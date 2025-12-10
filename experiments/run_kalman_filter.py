# experiments/analyze_kalman_stability.py
import numpy as np
import tensorflow as tf
from src.ssm import LinearGaussianSSM
from src.filter import Filter
from src.utility import max_asym, cond_matrix

def run():
    dx, dy = 3, 2
    dtype = tf.float32

    A = tf.constant([[0.9, 0.1, 0.0],
                     [0.0, 0.8, 0.1],
                     [0.0, 0.0, 0.9]], dtype=dtype)
    B = tf.eye(dx, dtype=dtype)
    C = tf.constant([[1.0, 0.5, 0.0],
                     [0.0, 0.9, 0.0]], dtype=dtype)
    D = 0.3 * tf.eye(dy, dtype=dtype)

    m0 = np.zeros(dx, dtype=np.float32)
    P0 = np.eye(dx, dtype=np.float32)

    ssm = LinearGaussianSSM(A=A, B=B, C=C, D=D, m0=m0, P0=P0)

    T = 100
    x_traj, y_traj = ssm.simulate(T=T, batch_size=1, seed=0)
    y = y_traj

    kalman = Filter(ssm)
    res_j = kalman.filter(y, joseph=True)
    res_s = kalman.filter(y, joseph=False)

    P_j = res_j["P_filt"][0]    # [T, dx, dx]
    P_s = res_s["P_filt"][0]
    S_j = res_j["S"][0]         # innovation covariances if you store them
    S_s = res_s["S"][0]

    # condition numbers per time step
    cond_P_j = cond_matrix(P_j)  # [T]
    cond_P_s = cond_matrix(P_s)
    cond_S_j = cond_matrix(S_j)
    cond_S_s = cond_matrix(S_s)

    print("=== Joseph vs Std: covariance conditioning ===")
    print("P (Joseph): min / max cond =", float(tf.reduce_min(cond_P_j)),
                                       "/", float(tf.reduce_max(cond_P_j)))
    print("P (Std)   : min / max cond =", float(tf.reduce_min(cond_P_s)),
                                       "/", float(tf.reduce_max(cond_P_s)))

    print("S (Joseph): min / max cond =", float(tf.reduce_min(cond_S_j)),
                                       "/", float(tf.reduce_max(cond_S_j)))
    print("S (Std)   : min / max cond =", float(tf.reduce_min(cond_S_s)),
                                       "/", float(tf.reduce_max(cond_S_s)))

    print("\nMax asymmetry ||P - P^T||:")
    print("Joseph:", float(max_asym(P_j)))
    print("Std   :", float(max_asym(P_s)))

if __name__ == "__main__":
    run()
