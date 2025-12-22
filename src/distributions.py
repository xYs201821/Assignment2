import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Proposal(tf.Module):
    def __init__(self):
        super().__init__()

    def dist(self, ssm, x_prev, y_t):
        raise NotImplementedError

    def sample(self, ssm, x_prev, y_t, seed=None):
        z = self.dist(ssm, x_prev, y_t).sample(seed=seed)
        return z, self.log_prob(ssm, z, x_prev, y_t)

    def log_prob(self, ssm, x, x_prev, y_t):
        return self.dist(ssm, x_prev, y_t).log_prob(x)


class BootstrapProposal(Proposal):
    def dist(self, ssm, x_prev, y_t):
        return ssm.transition_dist(x_prev)
