"""Shared beta-schedule utilities for particle flows."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

@dataclass(frozen=True)
class BetaScheduleConfig:
    """Configuration for beta schedule construction."""

    mode: str = "linear"
    mu: float = 0.2
    guard: str | bool | None = None


def _ensure_rank2(t: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    t = tf.convert_to_tensor(t, dtype=dtype)
    t_rank = tf.rank(t)
    tf.debugging.assert_rank_in(
        t,
        [1, 2],
        message="beta schedule tensors must be rank 1 or rank 2",
    )
    return tf.cond(tf.equal(t_rank, 1), lambda: t[tf.newaxis, :], lambda: t)


def linear_beta_schedule(
    num_lambda: int,
    dtype: tf.DType = tf.float32,
) -> tuple[tf.Tensor, tf.Tensor, float]:
    """Return a linear beta schedule using start-of-step values."""
    num_lambda = int(num_lambda)
    beta = tf.linspace(
        tf.cast(0.0, dtype),
        tf.cast(1.0, dtype),
        num_lambda + 1,
    )[:-1]
    beta = beta[tf.newaxis, :]
    beta_dot = tf.ones_like(beta)
    dl = 1.0 / float(num_lambda)
    return beta, beta_dot, dl


def build_beta_schedule(
    mode: str,
    num_lambda: int,
    dtype: tf.DType,
    P0_inv: tf.Tensor | None = None,
    Info: tf.Tensor | None = None,
    jitter: float | None = None,
    mu: float = 0.2,
) -> tuple[tf.Tensor, tf.Tensor, float]:
    """Build a beta schedule from mode and mu."""
    mode = "linear" if mode is None else str(mode).lower()

    if mode in ("linear", "lin"):
        beta, beta_dot, dl = linear_beta_schedule(num_lambda, dtype=dtype)
        return beta, beta_dot, dl
    if mode in ("optimal", "opt"):
        if P0_inv is None or Info is None:
            raise ValueError("P0_inv and Info are required to build an optimal beta schedule")
        solver = OptimalBetaSolver(num_lambda, jitter=jitter)
        beta, beta_dot, _ = solver.solve(
            P0_inv,
            Info,
            mu=mu,
            num_lambda=num_lambda,
        )
        return beta, beta_dot, 1.0 / float(num_lambda)
    raise ValueError("beta_schedule.mode must be 'linear' or 'optimal'")


def _cond_number_f(
    P0_inv: tf.Tensor,
    Info: tf.Tensor,
    Q: tf.Tensor | None,
    beta: tf.Tensor,
    beta_dot: tf.Tensor,
    dtype: tf.DType,
    jitter: float | None = None,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Compute condition number of F to avoid degradation."""
    P0_inv = tf.cast(P0_inv, dtype)
    Info = tf.cast(Info, dtype)
    beta = _ensure_rank2(beta, dtype)
    beta_dot = _ensure_rank2(beta_dot, dtype)
    beta_b = beta[..., tf.newaxis, tf.newaxis]
    beta_dot_b = beta_dot[..., tf.newaxis, tf.newaxis]

    M = P0_inv[:, tf.newaxis, :, :] + beta_b * Info[:, tf.newaxis, :, :]

    jitter_val = 1e-6 if jitter is None else float(jitter)
    eye = tf.eye(tf.shape(M)[-1], batch_shape=tf.shape(M)[:-2], dtype=dtype)
    M_sym = 0.5 * (M + tf.linalg.matrix_transpose(M)) + tf.cast(jitter_val, dtype) * eye
    L = tf.linalg.cholesky(M_sym)
    M_inv = tf.linalg.cholesky_solve(L, eye)

    Info_exp = Info[:, tf.newaxis, :, :]
    Minv_Info = tf.linalg.matmul(M_inv, Info_exp)

    if Q is None:
        Q_exp = tf.zeros_like(M)
    else:
        Q_exp = tf.broadcast_to(tf.cast(Q, dtype), tf.shape(M))
    F = -0.5 * (tf.linalg.matmul(Q_exp, M) + beta_dot_b * Minv_Info)
    s = tf.linalg.svd(F, compute_uv=False)
    s_max = tf.reduce_max(s, axis=-1)
    s_min = tf.reduce_min(s, axis=-1)
    return s_max / tf.maximum(s_min, tf.cast(eps, dtype))


class OptimalBetaSolver:
    """Solve the optimal beta schedule via shooting+bisection with RK4."""

    def __init__(self, num_lambda: int, jitter: float | None = 1e-6) -> None:
        self.num_lambda = int(num_lambda)
        self.jitter = jitter

    @tf.function
    def solve(
        self,
        P0_inv,
        Info,
        mu=0.2,
        num_lambda=None,
        tol=1e-6,
        max_bisect=50,
        max_bracket=30,
        v0_low=0.0,
        v0_high=1.0,
        beta_floor=0.0,
        target_beta=1.0,
        debug_shapes: bool = False,
    ):
        """Solve optimal beta schedule via shooting+bisection with RK4 inner integrator.

        Args:
          P0_inv: [batch, dx, dx] prior precision (no particle dimension).
          Info: [batch, dx, dx] information matrix (no particle dimension).
          mu: scalar coefficient in beta ODE.
          num_lambda: number of integration steps (defaults to self.num_lambda).
          tol: bisection tolerance on v0 interval.
          max_bisect: maximum bisection iterations.
          max_bracket: maximum bracket expansions for v0_high.
          v0_low/v0_high: initial bracket for beta_dot(0).
          beta_floor: optional lower bound for beta in M(beta) to keep SPD (>=0).
          target_beta: desired beta(1) target (default 1.0).
        Returns:
          beta: [batch, num_lambda] schedule (start-of-step beta values, float32).
          beta_dot: [batch, num_lambda] schedule (start-of-step beta_dot values, float32).
          v0_star: [batch] optimal initial beta_dot(0), float32.
        """
        compute_dtype = tf.float32
        P0_inv = tf.cast(P0_inv, compute_dtype)
        Info = tf.cast(Info, compute_dtype)
        mu = tf.cast(mu, compute_dtype)
        tol = tf.cast(tol, compute_dtype)
        target_beta = tf.cast(target_beta, compute_dtype)
        beta_floor = tf.cast(beta_floor, compute_dtype)

        tf.debugging.assert_rank(Info, 3)
        tf.debugging.assert_rank(P0_inv, 3)
        tf.debugging.assert_equal(tf.shape(Info)[0], tf.shape(P0_inv)[0])

        Info = 0.5 * (Info + tf.linalg.matrix_transpose(Info))
        P0_inv = 0.5 * (P0_inv + tf.linalg.matrix_transpose(P0_inv))

        if debug_shapes:
            tf.print(
                "solve_optimal_beta_schedule shapes",
                "P0_inv:", tf.shape(P0_inv),
                "Info:", tf.shape(Info),
                "num_lambda:", self.num_lambda,
            )

        num_steps = self.num_lambda if num_lambda is None else num_lambda
        num_steps = tf.cast(num_steps, tf.int32)
        h = tf.cast(1.0, compute_dtype) / tf.cast(num_steps, compute_dtype)

        trace_info = tf.linalg.trace(Info)
        v_lo = tf.cast(v0_low, compute_dtype) + tf.zeros_like(trace_info)
        v_hi = tf.cast(v0_high, compute_dtype) + tf.zeros_like(trace_info)

        def inverse_spd(M):
            jitter_val = 1e-6 if self.jitter is None else float(self.jitter)
            jitter_val = tf.cast(jitter_val, M.dtype)
            eye = tf.eye(tf.shape(M)[-1], batch_shape=tf.shape(M)[:-2], dtype=M.dtype)
            M_sym = 0.5 * (M + tf.linalg.matrix_transpose(M)) + jitter_val * eye
            L = tf.linalg.cholesky(M_sym)
            return tf.linalg.cholesky_solve(L, eye)

        def dkappa_dbeta(beta):
            beta = tf.cast(beta, compute_dtype)
            beta = tf.maximum(beta, beta_floor)  # prevent negative beta
            beta_b = beta[..., tf.newaxis, tf.newaxis]
            M_beta = P0_inv + beta_b * Info
            M_inv = inverse_spd(M_beta)
            tr_M = tf.linalg.trace(M_beta)
            tr_M_inv = tf.linalg.trace(M_inv)
            Minv_Info = tf.linalg.matmul(M_inv, Info)
            tr_minv_info_minv = tf.linalg.trace(tf.linalg.matmul(Minv_Info, M_inv))
            return trace_info * tr_M_inv - tr_M * tr_minv_info_minv

        def ode_rhs(beta, v):
            return v, mu * dkappa_dbeta(beta)

        def rk4_step(beta, v):
            # ode integrator
            k1b, k1v = ode_rhs(beta, v)
            k2b, k2v = ode_rhs(beta + 0.5 * h * k1b, v + 0.5 * h * k1v)
            k3b, k3v = ode_rhs(beta + 0.5 * h * k2b, v + 0.5 * h * k2v)
            k4b, k4v = ode_rhs(beta + h * k3b, v + h * k3v)
            beta_next = beta + (h / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
            v_next = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            beta_next = tf.maximum(beta_next, beta_floor)  # prevent negative beta
            return beta_next, v_next

        def integrate_endpoint(v0):
            beta = tf.zeros_like(v0)
            v = v0
            i0 = tf.constant(0, dtype=tf.int32)
            if debug_shapes:
                tf.print(
                    "integrate_endpoint entry",
                    "v0:", tf.shape(v0),
                    "beta:", tf.shape(beta),
                    "v:", tf.shape(v),
                )

            def cond(i, beta, v):
                return i < num_steps

            def body(i, beta, v):
                beta, v = rk4_step(beta, v)
                if debug_shapes:
                    def _print():
                        tf.print(
                            "integrate_endpoint step",
                            "i:", i,
                            "beta:", tf.shape(beta),
                            "v:", tf.shape(v),
                        )
                        return 0

                    _ = tf.cond(tf.equal(i, 0), _print, lambda: 0)
                return i + 1, beta, v

            _, beta, _ = tf.while_loop(cond, body, [i0, beta, v])
            return beta

        def residual(v0):
            return integrate_endpoint(v0) - target_beta

        # --- bracket v0_low until residual(v_lo) <= 0 ---
        r_lo = residual(v_lo)
        i0 = tf.constant(0, dtype=tf.int32)

        def bracket_lo_cond(i, v_lo, v_hi, r_lo):
            need = tf.reduce_any(r_lo > 0.0)
            return tf.logical_and(i < tf.cast(max_bracket, tf.int32), need)

        def bracket_lo_body(i, v_lo, v_hi, r_lo):
            step = tf.maximum(tf.abs(v_lo), tf.abs(v_hi))
            step = tf.maximum(step, tf.ones_like(step))
            v_lo_new = tf.where(r_lo > 0.0, v_lo - 2.0 * step, v_lo)
            r_lo_new = residual(v_lo_new)
            return i + 1, v_lo_new, v_hi, r_lo_new

        _, v_lo, v_hi, r_lo = tf.while_loop(
            bracket_lo_cond,
            bracket_lo_body,
            [i0, v_lo, v_hi, r_lo],
        )

        # --- bracket v0_high until residual(v_hi) >= 0 ---
        r_hi = residual(v_hi)
        i0 = tf.constant(0, dtype=tf.int32)

        def bracket_cond(i, v_lo, v_hi, r_hi):
            need = tf.reduce_any(r_hi < 0.0)
            return tf.logical_and(i < tf.cast(max_bracket, tf.int32), need)

        def bracket_body(i, v_lo, v_hi, r_hi):
            v_hi_new = tf.where(r_hi < 0.0, v_hi * 2.0, v_hi)
            r_hi_new = residual(v_hi_new)
            return i + 1, v_lo, v_hi_new, r_hi_new

        _, v_lo, v_hi, r_hi = tf.while_loop(bracket_cond, bracket_body, [i0, v_lo, v_hi, r_hi])

        # --- bisection on v0 (fallback to linear if bracket fails) ---
        r_lo = residual(v_lo)
        bracket_ok = tf.logical_and(
            tf.reduce_all(r_lo <= 0.0),
            tf.reduce_all(r_hi >= 0.0),
        )

        def _solve():
            k0 = tf.constant(0, dtype=tf.int32)

            def bisect_cond(k, v_lo, v_hi, r_lo, r_hi):
                width = v_hi - v_lo
                need = tf.reduce_any(width > tol)
                return tf.logical_and(k < tf.cast(max_bisect, tf.int32), need)

            def bisect_body(k, v_lo, v_hi, r_lo, r_hi):
                v_mid = 0.5 * (v_lo + v_hi)
                r_mid = residual(v_mid)
                use_hi = r_mid >= 0.0
                v_hi = tf.where(use_hi, v_mid, v_hi)
                r_hi = tf.where(use_hi, r_mid, r_hi)
                v_lo = tf.where(use_hi, v_lo, v_mid)
                r_lo = tf.where(use_hi, r_lo, r_mid)
                return k + 1, v_lo, v_hi, r_lo, r_hi

            _, v_lo_f, v_hi_f, _, _ = tf.while_loop(
                bisect_cond,
                bisect_body,
                [k0, v_lo, v_hi, r_lo, r_hi],
            )

            v0_star = 0.5 * (v_lo_f + v_hi_f)

            # --- integrate full trajectory with v0_star ---
            beta0 = tf.zeros_like(v0_star)
            v0 = v0_star
            ta_beta = tf.TensorArray(compute_dtype, size=num_steps + 1)
            ta_v = tf.TensorArray(compute_dtype, size=num_steps + 1)
            ta_beta = ta_beta.write(0, beta0)
            ta_v = ta_v.write(0, v0)

            def traj_cond(i, beta, v, ta_beta, ta_v):
                return i < num_steps

            def traj_body(i, beta, v, ta_beta, ta_v):
                beta, v = rk4_step(beta, v)
                if debug_shapes:
                    def _print():
                        tf.print(
                            "integrate_trajectory step",
                            "i:", i,
                            "beta:", tf.shape(beta),
                            "v:", tf.shape(v),
                        )
                        return 0

                    _ = tf.cond(tf.equal(i, 0), _print, lambda: 0)
                ta_beta = ta_beta.write(i + 1, beta)
                ta_v = ta_v.write(i + 1, v)
                return i + 1, beta, v, ta_beta, ta_v

            i0 = tf.constant(0, dtype=tf.int32)
            _, _, _, ta_beta, ta_v = tf.while_loop(
                traj_cond,
                traj_body,
                [i0, beta0, v0, ta_beta, ta_v],
            )

            beta_grid = ta_beta.stack()
            v_grid = ta_v.stack()

            beta_grid = tf.linalg.matrix_transpose(beta_grid)
            v_grid = tf.linalg.matrix_transpose(v_grid)
            beta = tf.cast(beta_grid[:, :-1], tf.float32)
            beta_dot = tf.cast(v_grid[:, :-1], tf.float32)
            v0_star_out = tf.cast(v0_star, tf.float32)
            return beta, beta_dot, v0_star_out

        def _fallback():
            beta_lin = tf.linspace(0.0, 1.0, num_steps + 1)[:-1]
            batch = tf.shape(P0_inv)[0]
            beta = tf.tile(beta_lin[tf.newaxis, :], [batch, 1])
            beta_dot = tf.ones_like(beta)
            v0_star = tf.ones([batch], dtype=tf.float32)
            return tf.cast(beta, tf.float32), tf.cast(beta_dot, tf.float32), v0_star

        return tf.cond(bracket_ok, _solve, _fallback)
