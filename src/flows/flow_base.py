import tensorflow as tf

from src.filters.particle import ParticleFilter
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.mixins import LinearizationMixin


class FlowBase(ParticleFilter, LinearizationMixin):
    """Shared scaffolding for EDH/LEDH flows."""

    def __init__(
        self,
        ssm,
        num_lambda=20,
        num_particles=100,
        ess_threshold=0.5,
        prior_stats="ekf",
        reweight="auto",
        init_from_particles="sample",
        debug=False,
        jitter=1e-12,
    ):
        super().__init__(ssm, num_particles=num_particles, ess_threshold=ess_threshold)
        self.num_lambda = int(num_lambda)
        self.prior_stats = self._validate_prior_stats(prior_stats)
        self.default_reweight = reweight
        self.init_from_particles = init_from_particles
        self._ekf = ExtendedKalmanFilter(ssm)
        self.debug = bool(debug)
        self.jitter = jitter
    @staticmethod
    def _validate_prior_stats(prior_stats):
        value = str(prior_stats)
        if value not in ("sample", "ekf"):
            raise ValueError(f"Invalid prior stats: {prior_stats}")
        return value

    # --- Prior builders (can be overridden by subclasses) ---
    def _prior_from_sample(self, mu_tilde, w, batch_shape, state_dim):
        m = tf.einsum("...n,...ni->...i", w, mu_tilde)
        mu_resid = mu_tilde - m[..., tf.newaxis, :]
        P = tf.einsum("...n,...ni,...nj->...ij", w, mu_resid, mu_resid)
        return m, P

    def _prior_from_ekf(self, m_prev, P_prev, batch_shape, batch_size, state_dim, q_dim):
        q0 = tf.zeros(tf.concat([batch_shape, [q_dim]], axis=0), dtype=tf.float32)
        q0_flat = tf.reshape(q0, tf.stack([batch_size, q_dim]))
        F_x_flat, m_pred_flat = self._jacobian(lambda x: self.ssm.f_with_noise(x, q0_flat), m_prev)
        F_q_flat, _ = self._jacobian(lambda q: self.ssm.f_with_noise(m_prev, q), q0_flat)
        F_x = tf.reshape(F_x_flat, tf.concat([batch_shape, tf.stack([state_dim, state_dim])], axis=0))
        F_q = tf.reshape(F_q_flat, tf.concat([batch_shape, tf.stack([state_dim, q_dim])], axis=0))
        P_pred = tf.einsum("...ij,...jk,...lk->...il", F_x, P_prev, F_x) + tf.einsum(
            "...ij,...jk,...lk->...il", F_q, self.ssm.cov_eps_x, F_q
        )
        return m_pred_flat, P_pred

    # --- Posterior builders (can be overridden by subclasses) ---
    def _posterior_from_sample(self, x_t, w_final):
        m = self.ssm.state_mean(x_t, w_final)
        P = self.ssm.state_cov(x_t, w_final)
        return {"m": m, "P": P}

    def _posterior_from_ekf(self, x_t, w_final, P_pred, y_t, batch_shape, batch_size, state_dim):
        m_flow = self.ssm.state_mean(x_t, w_final)
        obs_dim = tf.shape(y_t)[-1]
        m_flow_flat = tf.reshape(m_flow, tf.stack([batch_size, state_dim]))
        P_pred_flat = tf.reshape(P_pred, tf.stack([batch_size, state_dim, state_dim]))
        eye = tf.eye(state_dim, batch_shape=tf.shape(P_pred_flat)[:-2], dtype=P_pred_flat.dtype)
        P_pred_flat = tf.where(tf.math.is_finite(P_pred_flat), P_pred_flat, tf.zeros_like(P_pred_flat))
        P_pred_flat = P_pred_flat + eye * self.jitter
        y_flat = tf.reshape(y_t, tf.stack([batch_size, obs_dim]))
        m_filt_flat, P_filt_flat, _, _ = self._ekf.update(
            m_flow_flat,
            P_pred_flat,
            y_flat,
            joseph=True,
        )
        m_filt = tf.reshape(m_filt_flat, tf.concat([batch_shape, [state_dim]], axis=0))
        P_filt = tf.reshape(P_filt_flat, tf.concat([batch_shape, [state_dim, state_dim]], axis=0))
        return {"m": m_filt, "P": P_filt}

    # --- Abstract flow transport ---
    def _flow_transport(self, mu_tilde, y, m0, P):
        raise NotImplementedError

    # --- EKF tracker init (overridable) ---
    def _init_ekf_tracker(self, batch_shape, state_dim, tracker):
        if tracker is None:
            m0 = tf.convert_to_tensor(self.ssm.m0, dtype=tf.float32)
            P0 = tf.convert_to_tensor(self.ssm.P0, dtype=tf.float32)
            m_prev = tf.broadcast_to(m0, tf.concat([batch_shape, [state_dim]], axis=0))
            P_prev = tf.broadcast_to(P0, tf.concat([batch_shape, [state_dim, state_dim]], axis=0))
        else:
            m_prev = tracker["m"]
            P_prev = tracker["P"]
        return m_prev, P_prev

    def _propagate_particles(self, x_prev, seed=None):
        mu_tilde = self.ssm.sample_transition(x_prev, seed=seed)
        return mu_tilde, self.ssm.transition_dist(x_prev).log_prob(mu_tilde)

    def sample(self, x_prev, y_t, w=None, seed=None, m_prev=None, P_prev=None):
        batch_shape = tf.shape(x_prev)[:-2]
        batch_size = tf.reduce_prod(batch_shape)
        state_dim = tf.shape(x_prev)[-1]
        q_dim = tf.cast(self.ssm.q_dim, tf.int32)

        mu_tilde, log_q0 = self._propagate_particles(x_prev, seed=seed)
        if self.prior_stats == "sample":
            m, P = self._prior_from_sample(mu_tilde, w, batch_shape, state_dim)
        else:
            m, P = self._prior_from_ekf(m_prev, P_prev, batch_shape, batch_size, state_dim, q_dim)

        flow_out = self._flow_transport(mu_tilde, y_t, m, P)
        # flow_out contracts to (x_next, log_flow, m_aux, P_aux) for consistency
        if len(flow_out) == 2:
            x_next, log_flow = flow_out
            m_aux = P_aux = None
        else:
            x_next, m_aux, P_aux, log_flow = flow_out
        return x_next, log_q0 - log_flow, m_aux, P_aux, log_q0, log_flow

    @tf.function(reduce_retracing=True)
    def step(self, x_prev, log_w_prev, y_t, reweight="auto", tracker=None, **kwargs):
        if reweight is None:
            reweight = self.default_reweight

        batch_shape = tf.shape(y_t)[:-1]
        batch_size = tf.reduce_prod(batch_shape)
        state_dim = tf.cast(self.ssm.state_dim, tf.int32)

        if self.prior_stats == "ekf":
            m_prev, P_prev = self._init_ekf_tracker(batch_shape, state_dim, tracker)
            x_t, log_q, _, P_pred, log_q0, log_flow = self._step_with_prior(
                x_prev, y_t, m_prev, P_prev, batch_shape, batch_size, state_dim
            )
        else:
            x_t, log_q, _, _, log_q0, log_flow = self.sample(
                x_prev,
                y_t,
                w=tf.exp(log_w_prev),
                seed=self.ssm._tfp_seed(),
            )
            P_pred = None

        loglik = self.ssm.observation_dist(x_t).log_prob(y_t[..., tf.newaxis, :])
        logtrans = self.ssm.transition_dist(x_prev).log_prob(x_t)

        update_weights = reweight != 0
        if update_weights:
            log_w = log_w_prev + tf.cast(loglik + logtrans - log_q, tf.float32)
        else:
            log_w = log_w_prev

        log_w_norm, w, logZ = self._log_normalize(log_w)
        ess = self.ess(w)

        batch_shape_out = tf.shape(x_t)[:-2]
        parent_indices = tf.broadcast_to(
            tf.range(self.num_particles, dtype=tf.int32),
            tf.concat([batch_shape_out, [self.num_particles]], axis=0),
        )
        log_w_final = log_w_norm
        w_final = tf.exp(log_w_final)
        flow_info = {
            "ess": ess,
            "logZ": logZ,
            "log_w": log_w_norm,
            "parent_indices": parent_indices,
            "loglik": loglik,
            "logtrans": logtrans,
            "log_q": log_q,
            "log_q0": log_q0,
            "log_det": log_flow,
        }

        if self.prior_stats == "ekf":
            tracker = self._posterior_from_ekf(x_t, w_final, P_pred, y_t, batch_shape, batch_size, state_dim)
        else:
            tracker = self._posterior_from_sample(x_t, w_final)
        return x_t, log_w_final, w_final, tracker, flow_info

    def _step_with_prior(self, x_prev, y_t, m_prev, P_prev, batch_shape, batch_size, state_dim):
        q_dim = tf.cast(self.ssm.q_dim, tf.int32)
        m_pred, P_pred = self._prior_from_ekf(m_prev, P_prev, batch_shape, batch_size, state_dim, q_dim)
        mu_tilde, log_q0 = self._propagate_particles(x_prev, seed=self.ssm._tfp_seed())
        flow_out = self._flow_transport(mu_tilde, y_t, m_pred, P_pred)
        if len(flow_out) == 2:
            x_t, log_flow = flow_out
        else:
            x_t, _, _, log_flow = flow_out
        log_q = log_q0 - log_flow
        return x_t, log_q, m_pred, P_pred, log_q0, log_flow
