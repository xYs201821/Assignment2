import pytest
import tensorflow as tf


@pytest.mark.parametrize(
    "fixture_name",
    ["lgssm_2d", "lgssm_3d", "sv_model", "sv_model_logy2", "range_bearing_ssm"],
)
def test_distribution_shapes(request, fixture_name):
    ssm = request.getfixturevalue(fixture_name)
    batch_shape = (3,)

    x0_dist = ssm.initial_state_dist(batch_shape)
    x0 = x0_dist.sample()
    assert x0.shape == (batch_shape[0], ssm.state_dim)
    assert x0_dist.log_prob(x0).shape == (batch_shape[0],)

    trans_dist = ssm.transition_dist(x0)
    x1 = trans_dist.sample()
    assert x1.shape == x0.shape
    assert trans_dist.log_prob(x1).shape == (batch_shape[0],)

    obs_dist = ssm.observation_dist(x1)
    y1 = obs_dist.sample()
    assert y1.shape == (batch_shape[0], ssm.obs_dim)
    assert obs_dist.log_prob(y1).shape == (batch_shape[0],)
