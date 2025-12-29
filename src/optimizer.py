import tensorflow as tf

def update_norm(update):
    rank = tf.rank(update)
    return tf.cond(
        tf.equal(rank, 0),
        lambda: tf.abs(update),
        lambda: tf.norm(update, axis=-1),
    )

def apply_stop_mask(update, stopped):
    stopped = tf.convert_to_tensor(stopped, dtype=tf.bool)
    rank_diff = tf.rank(update) - tf.rank(stopped)
    ones = tf.ones(tf.stack([rank_diff]), dtype=tf.int32)
    new_shape = tf.concat([tf.shape(stopped), ones], axis=0)
    stopped = tf.reshape(stopped, new_shape)
    mask = tf.broadcast_to(stopped, tf.shape(update))
    return tf.where(mask, tf.zeros_like(update), update)

class FixStepSize:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def init_state(self, shape):
        # initialize state to zero
        return tf.zeros(shape, dtype=tf.float32)

    @tf.function
    def apply(self, x, grads, state):
        new_x = x - self.lr * grads
        return new_x, state

class FunctionalAdagrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        self.lr = learning_rate
        self.eps = tf.convert_to_tensor(epsilon, dtype=tf.float32)

    def init_state(self, shape):
        # initialize state to zero
        return tf.zeros(shape, dtype=tf.float32)

    @tf.function
    def apply(self, x, grads, state):
        """
        input: curr value, gradient, state
        """
        new_state = state + tf.square(grads)
        
        adjusted_grad = grads / (tf.sqrt(new_state) + self.eps)
        
        new_x = x - self.lr * adjusted_grad
        
        return new_x, new_state

class FunctionalAdam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = epsilon

    def init_state(self, shape):
        # adam: momentum and variance
        return (tf.zeros(shape), tf.zeros(shape), tf.constant(1.0)) # 最后一个是 step 计数

    @tf.function
    def apply(self, x, grads, state):
        m, v, t = state
        
        m_new = self.beta_1 * m + (1 - self.beta_1) * grads
        v_new = self.beta_2 * v + (1 - self.beta_2) * tf.square(grads)
        
        m_hat = m_new / (1 - tf.pow(self.beta_1, t))
        v_hat = v_new / (1 - tf.pow(self.beta_2, t))
        
        new_x = x - self.lr * m_hat / (tf.sqrt(v_hat) + self.eps)
        
        return new_x, (m_new, v_new, t + 1.0)

class StepControlledOptimizer:
    def __init__(self, optimizer, step_tol=None):
        self.optimizer = optimizer
        self.step_tol = step_tol

    def init_state(self, shape):
        base_state = self.optimizer.init_state(shape)
        if self.step_tol is None:
            return base_state
        stopped = tf.zeros(self._stopped_shape(shape), dtype=tf.bool)
        return base_state, stopped

    @staticmethod
    def _stopped_shape(shape):
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return []
            return list(shape[:-1])
        shape = tf.convert_to_tensor(shape)
        return shape[:-1]

    @tf.function
    def apply(self, x, grads, state):
        if self.step_tol is None:
            return self.optimizer.apply(x, grads, state)
        base_state, stopped = state
        new_x, new_state = self.optimizer.apply(x, grads, base_state)
        update = new_x - x
        update_norm_value = update_norm(update)
        stop_now = update_norm_value < tf.cast(self.step_tol, update_norm_value.dtype)
        stopped = tf.logical_or(stopped, stop_now)
        update = apply_stop_mask(update, stopped)
        new_x = x + update
        return new_x, (new_state, stopped)
