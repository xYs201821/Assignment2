import tensorflow as tf

from src.flows.flow_base import FlowBase
from src.optimizer import FixStepSize, FunctionalAdagrad, FunctionalAdam, apply_stop_mask, update_norm
from src.ssm import SSM
from src.utility import cholesky_solve, quadratic_matmul

class KernelParticleFlow(FlowBase):
    def __init__(
        self,
        ssm,
        num_particles=100,
        num_lambda=20,
        jitter=1e-6,
        ds_init=None,
        optimizer=None,
        optimizer_eps=1e-7,
        optimizer_beta_1=0.9,
        optimizer_beta_2=0.999,
        alpha=None,
        alpha_update_every=1,
        flow_tol=1e-6,
        kernel_type="diag",
        ll_grad_mode="linearized",
        localization_radius=None,
        max_flow_norm=None,
        debug=False,
        debug_every=1,
        assume_additive_obs_noise=True,
        ess_threshold=0.5,
        reweight="never",
    ):
        super().__init__(
            ssm,
            num_lambda=num_lambda,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            reweight=reweight,
            init_from_particles=False,
            debug=debug,
            jitter=jitter,
        )
        self.num_lambda = int(num_lambda)
        self.jitter = float(jitter)
        self.ds_init = 1.0 / float(self.num_lambda) if ds_init is None else float(ds_init)
        self.optimizer = self._build_optimizer(
            optimizer,
            optimizer_eps=optimizer_eps,
            optimizer_beta_1=optimizer_beta_1,
            optimizer_beta_2=optimizer_beta_2,
        )
        self.alpha = alpha
        self.alpha_update_every = int(alpha_update_every)
        if self.alpha_update_every < 1:
            self.alpha_update_every = 1
        self.flow_tol = flow_tol
        self.kernel_type = self._validate_kernel_type(kernel_type)
        if ll_grad_mode is None:
            ll_grad_mode = "linearized"
        self.ll_grad_mode = self._validate_ll_grad_mode(ll_grad_mode)
        self.localization_radius = localization_radius
        if max_flow_norm is None:
            self.max_flow_norm = None
        else:
            max_flow_norm = float(max_flow_norm)
            if max_flow_norm <= 0.0:
                raise ValueError("max_flow_norm must be positive")
            self.max_flow_norm = max_flow_norm
        self.assume_additive_obs_noise = bool(assume_additive_obs_noise)
        self._obs_noise_is_additive = (
            self.assume_additive_obs_noise
            and self.ssm.__class__.h_with_noise is SSM.h_with_noise
        )
        self._median_alpha = str(alpha).lower() == "median"
        if localization_radius is None:
            self.localization = None
        else:
            self.localization = self._build_localization_matrix(
                int(self.ssm.state_dim),
                localization_radius,
                dtype=tf.float32,
            )
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        if self.debug_every < 1:
            self.debug_every = 1

    @staticmethod
    def _validate_kernel_type(value):
        mode = str(value).lower()
        mapping = {
            "scalar": "scalar",
            "diag": "diag",
            "diagonal": "diag",
            "matrix_diag": "diag",
            "matrix_diagonal": "diag",
        }
        if mode not in mapping:
            raise ValueError("kernel_type must be 'scalar' or 'diag'")
        return mapping[mode]

    @staticmethod
    def _validate_ll_grad_mode(value):
        mode = str(value).lower()
        mapping = {
            "linearized": "linearized",
            "lin": "linearized",
            "dist": "dist",
            "distribution": "dist",
            "obs_dist": "dist",
            "observation_dist": "dist",
            "log_prob": "dist",
        }
        if mode not in mapping:
            raise ValueError("ll_grad_mode must be 'linearized' or 'dist'")
        return mapping[mode]

    @staticmethod
    def _normalize_optimizer_name(value):
        if value is None:
            return None
        name = str(value).lower()
        if name in ("none", "null", "false", "0"):
            return None
        return name

    def _build_optimizer(
        self,
        optimizer,
        optimizer_eps=1e-7,
        optimizer_beta_1=0.9,
        optimizer_beta_2=0.999,
    ):
        if optimizer is None:
            return None
        if hasattr(optimizer, "apply") and hasattr(optimizer, "init_state"):
            return optimizer
        name = self._normalize_optimizer_name(optimizer)
        if name is None:
            return None

        lr = self.ds_init
        eps = 1e-7 if optimizer_eps is None else float(optimizer_eps)
        if name in ("fixed", "fix", "sgd"):
            return FixStepSize(learning_rate=lr)
        if name in ("adagrad", "ada_grad"):
            return FunctionalAdagrad(learning_rate=lr, epsilon=eps)
        if name in ("adam",):
            beta_1 = 0.9 if optimizer_beta_1 is None else float(optimizer_beta_1)
            beta_2 = 0.999 if optimizer_beta_2 is None else float(optimizer_beta_2)
            return FunctionalAdam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=eps,
            )
        raise ValueError("optimizer must be one of: 'fixed', 'adagrad', 'adam'")

    @staticmethod
    def _build_localization_matrix(state_dim, radius, dtype=tf.float32):
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("localization_radius must be positive")
        idx = tf.range(state_dim, dtype=dtype)
        diff = idx[:, tf.newaxis] - idx[tf.newaxis, :]
        return tf.exp(-tf.square(diff) / tf.cast(radius, dtype))

    @staticmethod
    def _broadcast_observation(y_t, x):
        y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        obs_dim = tf.shape(y_t)[-1]
        target_shape = tf.concat([tf.shape(x)[:-1], tf.expand_dims(obs_dim, 0)], axis=0)
        return tf.broadcast_to(y_t[..., tf.newaxis, :], target_shape)

    def _use_additive_obs_noise(self):
        return self._obs_noise_is_additive and self.ssm.r_dim == self.ssm.obs_dim

    def _sample_mean_and_cov(self, x, w):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if w is None:
            w = tf.ones_like(x[..., 0]) / tf.cast(tf.shape(x)[-2], tf.float32)
        else:
            w = tf.convert_to_tensor(w, dtype=tf.float32)
        x_bar = tf.einsum("...n,...ni->...i", w, x)
        x_resid = x - x_bar[..., tf.newaxis, :]
        P = tf.einsum("...n,...ni,...nj->...ij", w, x_resid, x_resid)
        if self.localization is not None:
            P = P * tf.cast(self.localization, tf.float32)
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))
        I = tf.eye(tf.shape(P)[-1], batch_shape=tf.shape(P)[:-2], dtype=tf.float32)
        P = P + I * tf.cast(self.jitter, tf.float32)
        return x_bar, P

    def _prior_from_sample(self, mu_tilde, w):
        x_bar, P = self._sample_mean_and_cov(mu_tilde, w)
        return x_bar, P

    def _grad_log_prior_gaussian(self, x, x_bar, B):
        delta = x - x_bar[..., tf.newaxis, :] # [..., n, dx]
        rhs = tf.linalg.matrix_transpose(delta)
        sol = cholesky_solve(B, rhs, jitter=self.jitter)
        return -tf.linalg.matrix_transpose(sol)

    def _grad_log_likelihood_linearized(self, x, y):
        y_expand = self._broadcast_observation(y, x)
        r_dim = tf.cast(self.ssm.r_dim, tf.int32)
        batch_shape = tf.shape(x)[:-2]
        batch_size = tf.reduce_prod(batch_shape)
        state_dim = tf.shape(x)[-1]
        obs_dim = tf.shape(y)[-1]
        num_particles = tf.shape(x)[-2]
        r0 = tf.zeros(tf.concat([tf.shape(x)[:-1], [r_dim]], axis=0), dtype=tf.float32)

        r0_flat = tf.reshape(r0, tf.stack([batch_size * num_particles, r_dim]))
        x_flat = tf.reshape(x, tf.stack([batch_size * num_particles, state_dim]))
        H_x_flat, y_hat_flat = self.jacobian_h_x(x_flat, r0_flat)
        H_x = tf.reshape(
            H_x_flat,
            tf.concat([batch_shape, [num_particles, obs_dim, state_dim]], axis=0),
        )
        y_hat = tf.reshape(
            y_hat_flat,
            tf.concat([batch_shape, [num_particles, obs_dim]], axis=0),
        )

        v = self.ssm.innovation(y_expand, y_hat)  # [..., n, obs_dim]
        R = tf.convert_to_tensor(self.ssm.cov_eps_y, dtype=tf.float32)
        if self._use_additive_obs_noise():
            R_eff = tf.broadcast_to(
                R,
                tf.concat([batch_shape, tf.expand_dims(num_particles, 0), tf.shape(R)], axis=0),
            )
        else:
            H_r_flat, _ = self.jacobian_h_r(x_flat, r0_flat)
            H_r = tf.reshape(
                H_r_flat,
                tf.concat([batch_shape, [num_particles, obs_dim, r_dim]], axis=0),
            )
            R_eff = quadratic_matmul(H_r, R, H_r)
        z = cholesky_solve(R_eff, v[..., tf.newaxis], jitter=self.jitter)
        z = tf.squeeze(z, axis=-1)
        return tf.einsum("...nij,...ni->...nj", H_x, z)

    def _grad_log_likelihood_from_dist(self, x, y):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y_expand = self._broadcast_observation(y, x)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            loglik = self.ssm.observation_dist(x).log_prob(y_expand)
            loglik_sum = tf.reduce_sum(loglik)
        grad = tape.gradient(loglik_sum, x)
        if grad is None:
            raise ValueError(
                "observation_dist.log_prob has no gradients; set ll_grad_mode='linearized'"
            )
        return grad

    def _scalar_kernel_and_grad(self, x, band_inv, alpha=None):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        band_inv = tf.convert_to_tensor(band_inv, dtype=tf.float32)
        xj = x[..., :, tf.newaxis, :]
        xk = x[..., tf.newaxis, :, :]
        diff = xj - xk  # [..., n, n, dx]
        if alpha is None:
            alpha = self._compute_alpha(diff=diff)
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_inv = tf.math.reciprocal(alpha)
        # alpha_inv shape: [...], band_inv shape: [..., dx, dx]
        alpha_inv = alpha_inv[..., tf.newaxis, tf.newaxis]
        band_inv_scaled = band_inv * alpha_inv
        diff_scaled = tf.einsum("...ij,...mnj->...mni", band_inv_scaled, diff)
        q = tf.reduce_sum(diff_scaled * diff, axis=-1)
        K = tf.exp(-0.5 * q)
        # grad_K is dK/dx_k (second particle index); diff = x_j - x_k
        # K: [..., n, n], diff_scaled: [..., n, n, dx], grad_K: [..., n, n, dx]
        gradK = K[..., tf.newaxis] * diff_scaled
        return K, gradK

    def _kernel_and_grad(self, x, diagB, alpha):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        diagB = tf.convert_to_tensor(diagB, dtype=tf.float32)
        xj = x[..., :, tf.newaxis, :]
        xk = x[..., tf.newaxis, :, :]
        diff = xj - xk # [..., n, n, dx]

        if alpha is None:
            alpha = self._compute_alpha(diff=diff)
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_scalar = alpha
        # alpha_scalar shape: [...], diagB shape: [..., dx]
        alpha = alpha_scalar[..., tf.newaxis]
        alpha = tf.broadcast_to(alpha, tf.shape(diagB))
        alpha = alpha[..., tf.newaxis, tf.newaxis, :]
        denom = alpha * diagB[..., tf.newaxis, tf.newaxis, :] + tf.cast(self.jitter, tf.float32)
        K = tf.exp(-0.5 * tf.square(diff) / denom)  
        # grad_K is dK/dx_k (second particle index); diff = x_j - x_k
        # K: [..., n, n, dx], diff/denom: [..., n, n, dx], grad_K: [..., n, n, dx]
        gradK = (diff / denom) * K
        return K, gradK

    def _step_control(
        self,
        update,
        flow_norm_value,
        stopped,
        flow_tol,
    ):
        if flow_tol is None:
            update_eff = update
        else:
            tol = tf.cast(flow_tol, flow_norm_value.dtype)
            stop_now = flow_norm_value < tol
            stopped = tf.logical_or(stopped, stop_now)
            update_eff = apply_stop_mask(update, stopped)
        return update_eff, stopped

    def _debug_print(
        self,
        step_idx,
        flow_norm,
        grad_ll,
        grad_prior,
        x,
        diagB_mean,
        ds,
        flow_scale=None,
    ):
        if not self.debug:
            return
        if step_idx % self.debug_every != 0 and step_idx != self.num_lambda - 1:
            return
        grad_ll_norm = tf.reduce_mean(tf.norm(grad_ll, axis=-1), axis=-1)
        grad_prior_norm = tf.reduce_mean(tf.norm(grad_prior, axis=-1), axis=-1)
        ratio = tf.math.divide_no_nan(
            grad_ll_norm,
            grad_prior_norm + tf.cast(self.jitter, tf.float32),
        )
        x_mean_curr = tf.reduce_mean(x, axis=-2)
        if flow_scale is None:
            msg = tf.strings.format(
                "kflow iter={}/{} flow={} ds={} grad_ll={} grad_prior={} ratio={} x_mean={} diagB_mean={}",
                [
                    step_idx + 1,
                    self.num_lambda,
                    flow_norm,
                    ds,
                    grad_ll_norm,
                    grad_prior_norm,
                    ratio,
                    x_mean_curr,
                    diagB_mean,
                ],
            )
        else:
            msg = tf.strings.format(
                "kflow iter={}/{} flow={} scale={} ds={} grad_ll={} grad_prior={} ratio={} x_mean={} diagB_mean={}",
                [
                    step_idx + 1,
                    self.num_lambda,
                    flow_norm,
                    flow_scale,
                    ds,
                    grad_ll_norm,
                    grad_prior_norm,
                    ratio,
                    x_mean_curr,
                    diagB_mean,
                ],
            )
        tf.print(msg, summarize=8)

    def _flow_transport(self, mu_tilde, y, m0, P, w=None):
        x_next = self._pff_update(mu_tilde, y, w, x_mean=m0, B=P)
        # Kernel flow does not track Jacobian; treat log_det as zero.
        log_det = tf.zeros(tf.shape(mu_tilde)[:-1], dtype=tf.float32)
        flow_norm_max = tf.zeros(tf.shape(mu_tilde)[:-2], dtype=tf.float32)
        return x_next, log_det, flow_norm_max

    @staticmethod
    def _median_midpoint(x):
        shape = tf.shape(x)
        last_dim = shape[-1]
        batch = tf.reduce_prod(shape[:-1])
        flat = tf.reshape(x, tf.stack([batch, last_dim]))
        sorted_x = tf.sort(flat, axis=1)
        mid = last_dim // 2

        def even():
            upper = tf.gather(sorted_x, mid, axis=1)
            lower = tf.gather(sorted_x, mid - 1, axis=1)
            return (upper + lower) / 2.0

        def odd():
            return tf.gather(sorted_x, mid, axis=1)

        med = tf.cond(tf.equal(last_dim % 2, 0), even, odd)
        return tf.reshape(med, shape[:-1])

    @staticmethod
    def _median_midpoint_particles(x):
        shape = tf.shape(x)
        batch = tf.reduce_prod(shape[:-2])
        num_particles = shape[-2]
        state_dim = shape[-1]
        flat = tf.reshape(x, tf.stack([batch, num_particles, state_dim]))
        sorted_x = tf.sort(flat, axis=1)
        mid = num_particles // 2

        def even():
            upper = tf.gather(sorted_x, mid, axis=1)
            lower = tf.gather(sorted_x, mid - 1, axis=1)
            return (upper + lower) / 2.0

        def odd():
            return tf.gather(sorted_x, mid, axis=1)

        med = tf.cond(tf.equal(num_particles % 2, 0), even, odd)
        out_shape = tf.concat([shape[:-2], [state_dim]], axis=0)
        return tf.reshape(med, out_shape)

    def _compute_alpha(self, x=None, diff=None):
        if diff is None:
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            xj = x[..., :, tf.newaxis, :]
            xk = x[..., tf.newaxis, :, :]
            diff = xj - xk  # [..., n, n, dx]
        else:
            diff = tf.convert_to_tensor(diff, dtype=tf.float32)

        shape = tf.shape(diff)
        batch_shape = shape[:-3]
        batch = tf.reduce_prod(batch_shape)
        num_particles = shape[-3]
        diff_flat = tf.reshape(diff, tf.stack([batch, num_particles, num_particles, -1]))

        eye = tf.eye(num_particles, dtype=tf.int32)
        mask = tf.equal(eye, 0)
        mask = tf.reshape(mask, [num_particles * num_particles])

        def _alpha():
            dist = tf.norm(diff_flat, axis=-1)
            dist_flat = tf.reshape(dist, tf.stack([batch, num_particles * num_particles]))
            dist_pairs = tf.boolean_mask(dist_flat, mask, axis=1)
            med = self._median_midpoint(dist_pairs)
            return tf.square(med) / tf.math.log(tf.cast(self.ssm.state_dim, tf.float32))

        def _empty():
            return tf.zeros([batch], dtype=tf.float32)

        alpha = tf.cond(num_particles > 1, _alpha, _empty)
        alpha = tf.reshape(alpha, batch_shape)

        alpha = tf.maximum(tf.cast(alpha, tf.float32), tf.cast(self.jitter, tf.float32))
        return alpha

    def _pff_update(self, x_prior, y, w, x_mean=None, B=None):
        x = tf.identity(tf.convert_to_tensor(x_prior, dtype=tf.float32))
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        num_particles = tf.shape(x)[-2]
        batch_shape = tf.shape(x)[:-2]
        if w is None:
            w = tf.ones(tf.concat([batch_shape, [num_particles]], axis=0), dtype=tf.float32)
            w = w / tf.cast(num_particles, tf.float32)
        else:
            w = tf.convert_to_tensor(w, dtype=tf.float32)
            w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
        if x_mean is None or B is None:
            x_mean, B = self._sample_mean_and_cov(x_prior, w)
        B = tf.convert_to_tensor(B, dtype=tf.float32)
        diagB = tf.linalg.diag_part(B)
        diagB = tf.maximum(diagB, tf.cast(self.jitter, tf.float32))
        
        if self._median_alpha:
            alpha = None
        elif self.alpha is None:
            alpha = tf.cast(1.0, tf.float32) / tf.cast(num_particles, tf.float32)
        else:
            alpha = tf.cast(self.alpha, tf.float32)
        
        if self.kernel_type == "scalar":
            state_dim = tf.shape(x)[-1]
            eye = tf.eye(state_dim, batch_shape=tf.shape(B)[:-2], dtype=tf.float32)
            aB = B + tf.cast(self.jitter, tf.float32) * eye
            band_inv = cholesky_solve(aB, eye, jitter=self.jitter)
        else:
            band_inv = None
        opt_state = None
        if self.optimizer is not None:
            opt_state = self.optimizer.init_state(tf.shape(x))
        ds = tf.ones(batch_shape, dtype=tf.float32) * tf.cast(self.ds_init, tf.float32)
        diagB_mean = None
        if self.debug:
            diagB_mean = tf.reduce_mean(diagB, axis=-1)
        flow_tol = None if self.flow_tol is None else tf.cast(self.flow_tol, tf.float32)
        max_flow_norm = None if self.max_flow_norm is None else tf.cast(self.max_flow_norm, tf.float32)
        stopped = tf.zeros(batch_shape, dtype=tf.bool)
        for i in range(self.num_lambda):
            if self.ll_grad_mode == "dist":
                grad_ll = self._grad_log_likelihood_from_dist(x, y)
            else:
                grad_ll = self._grad_log_likelihood_linearized(x, y)
            grad_prior = self._grad_log_prior_gaussian(x, x_mean, B) # [..., n, dx]
            g = grad_prior + grad_ll
            alpha_step = alpha
            if self._median_alpha and (i % self.alpha_update_every != 0):
                alpha_step = None
            if self.kernel_type == "scalar":
                K, grad_K = self._scalar_kernel_and_grad(x, band_inv, alpha_step)
                # K: [..., n, n], grad_K: [..., n, n, dx], g: [..., n, dx]
                term1 = K[..., tf.newaxis] * g[..., tf.newaxis, :, :]
            else:
                K, grad_K = self._kernel_and_grad(x, diagB, alpha_step)
                # K: [..., n, n, dx], grad_K: [..., n, n, dx], g: [..., n, dx]
                term1 = K * g[..., tf.newaxis, :, :]
            weighted = w[..., tf.newaxis, :, tf.newaxis] * (term1 + grad_K)
            flow = tf.reduce_sum(weighted, axis=-2)
            flow = tf.einsum("...ij,...nj->...ni", B, flow) # preconditioning with B
            flow_norm_particles = update_norm(flow)
            flow_norm = tf.reduce_mean(flow_norm_particles, axis=-1)
            if max_flow_norm is not None:
                flow_scale = tf.math.divide_no_nan(
                    max_flow_norm,
                    flow_norm + tf.cast(self.jitter, tf.float32),
                )
                flow_scale = tf.minimum(flow_scale, 1.0)
                flow = flow * flow_scale[..., tf.newaxis, tf.newaxis]
                flow_norm = flow_norm * flow_scale
            self._debug_print(
                i,
                flow_norm,
                grad_ll,
                grad_prior,
                x,
                diagB_mean,
                ds,
                flow_scale if max_flow_norm is not None else None,
            )
            if self.optimizer is None:
                update = ds[..., tf.newaxis, tf.newaxis] * flow
            else:
                new_x, opt_state = self.optimizer.apply(x, -flow, opt_state)
                update = new_x - x
            update_eff, stopped = self._step_control(
                update,
                flow_norm,
                stopped,
                flow_tol,
            )
            x = x + update_eff
        return x

    @tf.function
    def step(
        self,
        x_prev,
        log_w_prev,
        y_t,
        reweight="auto",
        **kwargs,
    ):
        if reweight is None:
            reweight = self.default_reweight

        mu_tilde = self.ssm.sample_transition(x_prev, seed=self.ssm._tfp_seed())
        trans_dist = self.ssm.transition_dist(x_prev)
        log_q0 = trans_dist.log_prob(mu_tilde)

        w_prev = tf.exp(log_w_prev)
        w_prev = tf.math.divide_no_nan(w_prev, tf.reduce_sum(w_prev, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w_prev)
        P_pred = self.ssm.state_cov(mu_tilde, w_prev)

        m, P = self._prior_from_sample(mu_tilde, w_prev)
        x_t, log_det, _ = self._flow_transport(mu_tilde, y_t, m, P, w=w_prev)

        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])

        update_weights = reweight != 0
        if update_weights:
            log_prior = trans_dist.log_prob(x_t)
            log_q = log_q0 - log_det
            log_w = log_w_prev + tf.cast(loglik + log_prior - log_q, tf.float32)
        else:
            log_w = log_w_prev

        log_w_norm, w, _ = self._log_normalize(log_w)

        batch_shape_out = tf.shape(x_t)[:-2]
        parent_indices = tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape_out, [self.num_particles]], axis=0),
        )
        return mu_tilde, x_t, log_w_norm, w, parent_indices, m_pred, P_pred

    def sample(self, x_prev, y_t, w=None, seed=None):
        mu_tilde, log_q0 = self._propagate_particles(x_prev, seed=seed)
        batch_shape = tf.shape(mu_tilde)[:-2]
        num_particles = tf.shape(mu_tilde)[-2]
        if w is None:
            w = tf.ones(tf.concat([batch_shape, [num_particles]], axis=0), dtype=tf.float32)
            w = w / tf.cast(num_particles, tf.float32)
        else:
            w = tf.convert_to_tensor(w, dtype=tf.float32)
            w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
        m_pred = self.ssm.state_mean(mu_tilde, w)
        P_pred = self.ssm.state_cov(mu_tilde, w)
        m0, P0 = self._prior_from_sample(mu_tilde, w)
        x_next, log_det, _ = self._flow_transport(mu_tilde, y_t, m0, P0, w=w)
        return x_next, log_q0 - log_det, m_pred, P_pred
    
