import tensorflow as tf
import numpy as np
from src.utility import tf_cond, cholesky_solve

class Filter(tf.Module):
    def __init__(self, ssm):
        super().__init__()
        self.ssm = ssm
        self.cov_eps_x = tf.linalg.matmul(self.ssm.B, self.ssm.B, adjoint_b=True)
        self.cov_eps_y = tf.linalg.matmul(self.ssm.D, self.ssm.D, adjoint_b=True)

    def predict(self, m_prev, P_prev, A, Q):
        m_pred = tf.einsum('ij,bj->bi', A, m_prev) # [batch, dx]
        P_pred = tf.einsum('ij,bjk,lk->bil', A, P_prev, A) + Q # [batch, dx, dx]
        return m_pred, P_pred # [batch, dx], [batch, dx, dx]

    def update(self, m_pred, P_pred, y, C, R, joseph=True):
        y_pred = tf.einsum('ij,bj->bi', C, m_pred)
        v = y - y_pred

        S = tf.einsum('ij,bjk,lk->bil', C, P_pred, C) + R # [batch, dy, dy]
        RHS = tf.einsum('bij,kj->bik', P_pred, C) 
        RHS_transpose = tf.transpose(RHS, perm=[0, 2, 1])
        
        K_transpose = cholesky_solve(S, RHS_transpose) # S K' = CP'
        K = tf.transpose(K_transpose, perm=[0, 2, 1])
        m_filt = m_pred + tf.einsum('bij,bj->bi', K, v) # [batch, dx]
        if joseph:
            # Joseph form: P_filt = (I - KC) P_pred (I - KC)' + K R K'
            I = tf.eye(tf.shape(P_pred)[1], batch_shape=[tf.shape(P_pred)[0]], dtype=tf.float32)
            I_KC = I - tf.einsum('bij,jk->bik', K, C)
            P_filt = tf.einsum('bij,bjk,blk->bil', I_KC, P_pred, I_KC) + tf.einsum('bij,jk,blk->bil', K, R, K)
        else:
            # standard form: P_filt = P_pred - K S K'
            P_filt = P_pred - tf.einsum('bij,bjk,blk->bil', K, S, K)
        
        cond_P = tf_cond(P_filt)
        cond_S = tf_cond(S)
        return m_filt, P_filt, cond_P, cond_S # [batch, dx], [batch, dx, dx], [batch], [batch]

    def filter(self, y, joseph=True): # y: [batch, T, dy]
        
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if (len(y.shape) == 2):
            y = y[tf.newaxis, :]
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]
        dx = self.ssm.state_dim
        dy = self.ssm.obs_dim

        A = self.ssm.A
        Q = self.cov_eps_x
        C = self.ssm.C
        R = self.cov_eps_y
        
        m_pred = tf.broadcast_to(self.ssm.m0, (batch_size, dx))
        P_pred = tf.broadcast_to(self.ssm.P0, (batch_size, dx, dx))

        m_filt_list = []
        P_filt_list = []
        m_pred_list = []
        P_pred_list = []
        cond_P_list = []
        cond_S_list = []
        
        for t in range(T):
            y_t = y[:, t, :]
            m_pred_list.append(m_pred)
            P_pred_list.append(P_pred)
            m_filt, P_filt, cond_P, cond_S = self.update(m_pred, P_pred, y_t, C, R, joseph) # [batch, dx], [batch, dx, dx]
            m_filt_list.append(m_filt)
            P_filt_list.append(P_filt)
            m_pred, P_pred = self.predict(m_filt, P_filt, A, Q) # [batch, dx], [batch, dx, dx]
            cond_P_list.append(cond_P)
            cond_S_list.append(cond_S)
        m_filt = tf.stack(m_filt_list, axis=1)   # [batch, T, dx]
        P_filt = tf.stack(P_filt_list, axis=1)   # [batch, T, dx, dx]
        m_pred = tf.stack(m_pred_list, axis=1)   # [batch, T, dx]
        P_pred = tf.stack(P_pred_list, axis=1)   # [batch, T, dx, dx]
        cond_P = tf.stack(cond_P_list, axis=1)   # [batch, T]
        cond_S = tf.stack(cond_S_list, axis=1)   # [batch, T]
        return {
            "m_filt": m_filt,
            "P_filt": P_filt,
            "m_pred": m_pred,
            "P_pred": P_pred,
            "cond_P": cond_P,
            "cond_S": cond_S,
        }
