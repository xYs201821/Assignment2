from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from src.utility import cholesky_solve, quadratic_matmul


def _to_seed(seed: tf.Tensor | int) -> tf.Tensor:
    seed_t = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed_t.shape.rank == 0:
        return tf.stack([seed_t, seed_t + tf.constant(1, tf.int32)], axis=0)
    seed_t = tf.reshape(seed_t, [-1])
    if tf.shape(seed_t)[0] >= 2:
        return tf.stack([seed_t[0], seed_t[1]], axis=0)
    return tf.stack([seed_t[0], seed_t[0] + tf.constant(1, tf.int32)], axis=0)


def _weighted_mean_cov(x: tf.Tensor, w: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    w = tf.math.divide_no_nan(w, tf.reduce_sum(w, axis=-1, keepdims=True))
    m = tf.einsum("bn,bnd->bd", w, x)
    r = x - m[:, tf.newaxis, :]
    p = tf.einsum("bn,bni,bnj->bij", w, r, r)
    return m, p


def _broadcast_logdet(logdet: tf.Tensor, log_q0: tf.Tensor) -> tf.Tensor:
    rank_diff = tf.rank(log_q0) - tf.rank(logdet)

    def _pad():
        shape = tf.concat([tf.shape(logdet), tf.ones(rank_diff, dtype=tf.int32)], axis=0)
        return tf.reshape(logdet, shape)

    return tf.cond(rank_diff > 0, _pad, lambda: logdet)


@dataclass
class PureBootstrapProposal:
    def sample(
        self,
        ssm,
        x_prev: tf.Tensor,
        y_t: tf.Tensor,
        *,
        params: dict[str, tf.Tensor],
        seed: tf.Tensor | int,
        y_prev: tf.Tensor | None = None,
        log_w_prev: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del y_t, log_w_prev
        dist = ssm.transition_dist(x_prev, y_prev=y_prev, params=params)
        x = tf.cast(dist.sample(seed=_to_seed(seed)), tf.float32)
        log_q = tf.cast(dist.log_prob(x), tf.float32)
        return x, log_q


def _edh_flow_solution(lam, h, p, r, y_tilde, m0, jitter=1e-5):
    hph = quadratic_matmul(h, p, h)
    lam_b = lam[..., tf.newaxis, tf.newaxis]
    s = lam_b * hph + r
    eye = tf.eye(tf.shape(s)[-1], batch_shape=tf.shape(s)[:-2], dtype=s.dtype)
    s = s + tf.cast(jitter, s.dtype) * eye
    rhs = tf.linalg.matmul(h, p, transpose_b=True)
    k_t = cholesky_solve(s, rhs, jitter=jitter)
    k = tf.linalg.matrix_transpose(k_t)
    a = -0.5 * tf.linalg.matmul(k, h)
    i = tf.eye(tf.shape(a)[-1], batch_shape=tf.shape(a)[:-2], dtype=a.dtype)
    b = tf.einsum("bij,bjk,bk->bi", i + lam_b * a, k, y_tilde)
    am0 = tf.einsum("bij,bjk,bk->bi", i + 2.0 * lam_b * a, a, m0)
    return a, b + am0


def _ledh_flow_solution(lam, h, p, r, y_tilde, m0, jitter=1e-5):
    hph = quadratic_matmul(h, p, h)
    lam_b = lam[..., tf.newaxis, tf.newaxis]
    s = lam_b * hph + r
    eye = tf.eye(tf.shape(s)[-1], batch_shape=tf.shape(s)[:-2], dtype=s.dtype)
    s = s + tf.cast(jitter, s.dtype) * eye
    pht = tf.linalg.matmul(p, h, transpose_b=True)
    k_t = cholesky_solve(s, tf.linalg.matrix_transpose(pht), jitter=jitter)
    k = tf.linalg.matrix_transpose(k_t)
    a = -0.5 * tf.linalg.matmul(k, h)
    i = tf.eye(tf.shape(a)[-1], batch_shape=tf.shape(a)[:-2], dtype=a.dtype)
    b = tf.einsum("bnij,bnjk,bnk->bni", i + lam_b * a, k, y_tilde)
    am0 = tf.einsum("bnij,bnjk,bnk->bni", i + 2.0 * lam_b * a, a, m0)
    return a, b + am0


@dataclass
class PureEDHProposal:
    num_lambda: int = 20
    jitter: float = 1e-5

    def sample(
        self,
        ssm,
        x_prev: tf.Tensor,
        y_t: tf.Tensor,
        *,
        params: dict[str, tf.Tensor],
        seed: tf.Tensor | int,
        y_prev: tf.Tensor | None = None,
        log_w_prev: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        seeds = tf.random.experimental.stateless_split(_to_seed(seed), 2)
        trans_dist = ssm.transition_dist(x_prev, y_prev=y_prev, params=params)
        mu = tf.cast(trans_dist.sample(seed=seeds[0]), tf.float32)
        log_q0 = tf.cast(trans_dist.log_prob(mu), tf.float32)
        if log_w_prev is None:
            w_prev = tf.ones_like(log_q0) / tf.cast(tf.shape(log_q0)[-1], tf.float32)
        else:
            lw = tf.cast(log_w_prev, tf.float32)
            lw = lw - tf.reduce_logsumexp(lw, axis=-1, keepdims=True)
            w_prev = tf.exp(lw)
        m0, p = _weighted_mean_cov(mu, w_prev)
        m_bar = tf.identity(m0)
        batch = tf.shape(mu)[0]
        n = tf.shape(mu)[1]
        dx = tf.shape(mu)[2]
        i = tf.eye(dx, batch_shape=[batch], dtype=mu.dtype)
        r_dim = tf.cast(ssm.r_dim, tf.int32)
        r0 = tf.zeros([batch, r_dim], dtype=mu.dtype)
        r_cov = ssm.observation_cov(params=params)
        step = tf.cast(1.0 / float(self.num_lambda), mu.dtype)
        logdet = tf.zeros([batch], dtype=tf.float32)

        for j in range(self.num_lambda):
            lam = tf.cast(j, mu.dtype) * step
            h, h_m = ssm.jacobian_h_x(m_bar, r0, params=params)
            h_r, _ = ssm.jacobian_h_r(m_bar, r0, params=params)
            hm = tf.einsum("bij,bj->bi", h, m_bar)
            y_tilde = ssm.innovation(y_t, h_m) + hm
            r_eff = quadratic_matmul(h_r, r_cov, h_r)
            a, b = _edh_flow_solution(lam, h, p, r_eff, y_tilde, m0, self.jitter)
            jmat = i + step[..., tf.newaxis, tf.newaxis] * a
            jmat = jmat + tf.cast(self.jitter, jmat.dtype) * i
            sign, lad = tf.linalg.slogdet(jmat)
            bad = tf.logical_or(tf.equal(sign, 0.0), tf.logical_not(tf.math.is_finite(lad)))
            lad = tf.where(bad, tf.zeros_like(lad), lad)
            logdet = logdet + lad
            am = tf.einsum("bij,bj->bi", a, m_bar)
            m_next = m_bar + step[..., tf.newaxis] * (am + b)
            ax = tf.einsum("bij,bnj->bni", a, mu)
            mu_next = mu + step[..., tf.newaxis, tf.newaxis] * (ax + b[:, tf.newaxis, :])
            m_bar = tf.where(bad[:, tf.newaxis], m_bar, m_next)
            mu = tf.where(bad[:, tf.newaxis, tf.newaxis], mu, mu_next)

        log_q = log_q0 - _broadcast_logdet(logdet, log_q0)
        return mu, tf.cast(log_q, tf.float32)


@dataclass
class PureLEDHProposal:
    num_lambda: int = 20
    jitter: float = 1e-5

    def sample(
        self,
        ssm,
        x_prev: tf.Tensor,
        y_t: tf.Tensor,
        *,
        params: dict[str, tf.Tensor],
        seed: tf.Tensor | int,
        y_prev: tf.Tensor | None = None,
        log_w_prev: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        seeds = tf.random.experimental.stateless_split(_to_seed(seed), 2)
        trans_dist = ssm.transition_dist(x_prev, y_prev=y_prev, params=params)
        mu = tf.cast(trans_dist.sample(seed=seeds[0]), tf.float32)
        log_q0 = tf.cast(trans_dist.log_prob(mu), tf.float32)
        if log_w_prev is None:
            w_prev = tf.ones_like(log_q0) / tf.cast(tf.shape(log_q0)[-1], tf.float32)
        else:
            lw = tf.cast(log_w_prev, tf.float32)
            lw = lw - tf.reduce_logsumexp(lw, axis=-1, keepdims=True)
            w_prev = tf.exp(lw)
        _, p = _weighted_mean_cov(mu, w_prev)
        batch = tf.shape(mu)[0]
        n = tf.shape(mu)[1]
        dx = tf.shape(mu)[2]
        r_dim = tf.cast(ssm.r_dim, tf.int32)
        obs_dim = tf.shape(y_t)[-1]
        i = tf.eye(dx, batch_shape=[batch, n], dtype=mu.dtype)
        p_exp = tf.broadcast_to(p[:, tf.newaxis, :, :], [batch, n, dx, dx])
        r_cov = ssm.observation_cov(params=params)
        r_exp = tf.broadcast_to(r_cov[tf.newaxis, tf.newaxis, :, :], [batch, n, obs_dim, obs_dim])
        y_exp = tf.broadcast_to(y_t[:, tf.newaxis, :], [batch, n, obs_dim])
        r0 = tf.zeros([batch, n, r_dim], dtype=mu.dtype)
        step = tf.cast(1.0 / float(self.num_lambda), mu.dtype)
        logdet = tf.zeros([batch, n], dtype=tf.float32)

        for j in range(self.num_lambda):
            lam = tf.cast(j, mu.dtype) * step
            lam_bn = tf.ones([batch, n], dtype=mu.dtype) * lam
            h, h_loc = ssm.jacobian_h_x(mu, r0, params=params)
            h_r, _ = ssm.jacobian_h_r(mu, r0, params=params)
            hx = tf.einsum("bnij,bnj->bni", h, mu)
            y_tilde = ssm.innovation(y_exp, h_loc) + hx
            r_eff = quadratic_matmul(h_r, r_exp, h_r)
            a, b = _ledh_flow_solution(lam_bn, h, p_exp, r_eff, y_tilde, mu, self.jitter)
            jmat = i + step[..., tf.newaxis, tf.newaxis] * a
            jmat = jmat + tf.cast(self.jitter, jmat.dtype) * i
            sign, lad = tf.linalg.slogdet(jmat)
            bad = tf.logical_or(tf.equal(sign, 0.0), tf.logical_not(tf.math.is_finite(lad)))
            lad = tf.where(bad, tf.zeros_like(lad), lad)
            logdet = logdet + lad
            ax = tf.einsum("bnij,bnj->bni", a, mu)
            mu_next = mu + step[..., tf.newaxis] * (ax + b)
            mu = tf.where(bad[..., tf.newaxis], mu, mu_next)

        log_q = log_q0 - _broadcast_logdet(logdet, log_q0)
        return mu, tf.cast(log_q, tf.float32)


def build_pure_proposal(kind: str, num_lambda: int) -> object:
    k = str(kind).strip().lower()
    if k == "bootstrap":
        return PureBootstrapProposal()
    if k == "ledh":
        return PureLEDHProposal(num_lambda=int(num_lambda))
    if k == "edh":
        return PureEDHProposal(num_lambda=int(num_lambda))
    raise ValueError("proposal_kind must be one of {'bootstrap', 'ledh', 'edh'}")

