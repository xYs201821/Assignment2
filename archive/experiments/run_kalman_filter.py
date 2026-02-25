# experiments/analyze_kalman_stability.py
import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.filter import KalmanFilter
from src.utility import max_asym
from experiments.lgssm_utils import build_lgssm
def run_lgssm_experiment():
    ssm, _, _ = build_lgssm(seed=42)

    T = 100
    x_traj, y_traj = ssm.simulate(T=T, shape=(1, ))
    y = y_traj

    kalman = KalmanFilter(ssm)
    res = kalman.filter(y, joseph=True)

    P_filt = res["P_filt"][0]    # [T, dx, dx]
    P_pred = res["P_pred"][0]
    m_filt = res["m_filt"][0]
    m_pred = res["m_pred"][0]
    cond_P = res["cond_P"][0]
    cond_S = res["cond_S"][0]
    # mse error
    mse_filt = tf.reduce_mean((m_filt - x_traj) ** 2)
    mse_pred = tf.reduce_mean((m_pred - x_traj) ** 2)
    print("mse_filt:", mse_filt)
    print("mse_pred:", mse_pred)
    # condition numbers per time step
    print("P (Joseph): min / max cond =", float(tf.reduce_min(cond_P)),
                                       "/", float(tf.reduce_max(cond_P)))
    print("S (Joseph): min / max cond =", float(tf.reduce_min(cond_S)),
                                       "/", float(tf.reduce_max(cond_S)))

    print("\nMax asymmetry ||P - P^T||:")
    print("Joseph:", float(max_asym(P_filt)))
    print("Std   :", float(max_asym(P_pred)))

if __name__ == "__main__":
    run_lgssm_experiment()
