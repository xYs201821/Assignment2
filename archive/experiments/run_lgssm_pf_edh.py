import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.filters.particle import BootstrapParticleFilter
from src.filter import KalmanFilter
from src.flows.edh import EDHFlow
from src.utility import weighted_mean, tfp_lgssm
from experiments.lgssm_utils import build_lgssm


def rmse(x, y):
    return tf.sqrt(tf.reduce_mean(tf.square(x - y)))


def run():
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    T = 80
    batch_size = 1
    num_particles = 1000
    num_lambda = 20
    warmup = 10

    ssm, _, _ = build_lgssm(seed=seed)
    x_traj, y_traj = ssm.simulate(T=T, shape=(batch_size,))
    x_true = x_traj[0]
    y = y_traj

    m_tfp, _ = tfp_lgssm(y[0], ssm, mode="filter")
    m_ref = m_tfp[:T]

    kalman = KalmanFilter(ssm)
    kf_out = kalman.filter(y, joseph=True)
    x_mean_kf = kf_out["m_filt"][0]

    ssm.set_seed(seed)
    pf = BootstrapParticleFilter(ssm, num_particles=num_particles, ess_threshold=0.5)
    x_particles_pf, w_pf, _, _ = pf.filter(y, reweight="always")
    x_mean_pf = weighted_mean(x_particles_pf, w_pf, axis=-2)[0]

    ssm.set_seed(seed)
    edh = EDHFlow(ssm, num_lambda=num_lambda, num_particles=num_particles, ess_threshold=0.5)
    x_particles_edh, w_edh, _, _ = edh.filter(y, reweight="always")
    x_mean_edh = weighted_mean(x_particles_edh, w_edh, axis=-2)[0]

    print("LGSSM vs x_true on mean state")
    print("  T =", T, "N =", num_particles, "num_lambda =", num_lambda, "warmup =", warmup)
    print("  KF RMSE :", float(rmse(x_mean_kf, x_true)))
    print("  PF RMSE :", float(rmse(x_mean_pf, x_true)))
    print("  EDH RMSE:", float(rmse(x_mean_edh, x_true)))
    print("  KF max abs :", float(tf.reduce_max(tf.abs(x_mean_kf[-1] - x_true[-1]))))
    print("  PF max abs :", float(tf.reduce_max(tf.abs(x_mean_pf[-1] - x_true[-1]))))
    print("  EDH max abs:", float(tf.reduce_max(tf.abs(x_mean_edh[-1] - x_true[-1]))))
    print("  TFP RMSE :", float(rmse(m_ref, x_true)))
    print("  TFP max abs :", float(tf.reduce_max(tf.abs(m_ref[-1] - x_true[-1]))))
    print("  KF vs PF RMSE :", float(rmse(x_mean_kf, x_mean_pf)))
    print("  KF vs EDH RMSE:", float(rmse(x_mean_kf, x_mean_edh)))
    print("  PF vs EDH RMSE:", float(rmse(x_mean_pf, x_mean_edh)))

    if warmup > 0 and warmup < T:
        x_true_w = x_true[warmup:]
        x_mean_kf_w = x_mean_kf[warmup:]
        x_mean_pf_w = x_mean_pf[warmup:]
        x_mean_edh_w = x_mean_edh[warmup:]
        print("Warm-up excluded RMSE (t >= warmup)")
        print("  KF RMSE :", float(rmse(x_mean_kf_w, x_true_w)))
        print("  PF RMSE :", float(rmse(x_mean_pf_w, x_true_w)))
        print("  EDH RMSE:", float(rmse(x_mean_edh_w, x_true_w)))
        m_ref_w = m_ref[warmup:]
        print("  TFP RMSE :", float(rmse(m_ref_w, x_true_w)))


if __name__ == "__main__":
    run()
