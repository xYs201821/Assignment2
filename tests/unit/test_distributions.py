import pytest
import tensorflow as tf

from tests.testhelper import assert_all_finite

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "fixture_name",
    ["lgssm_2d", "lgssm_3d", "sv_model", "sv_model_logy2", "range_bearing_ssm"],
)
def test_distribution_shapes_and_logprob(request, fixture_name):
    ssm = request.getfixturevalue(fixture_name)
    batch_shape = (3,)

    x0_dist = ssm.initial_state_dist(batch_shape)
    x0 = x0_dist.sample()
    logp0 = x0_dist.log_prob(x0)
    assert x0.shape == (batch_shape[0], ssm.state_dim)
    assert logp0.shape == (batch_shape[0],)
    assert_all_finite(x0, logp0)

    trans_dist = ssm.transition_dist(x0)
    x1 = trans_dist.sample()
    logp1 = trans_dist.log_prob(x1)
    assert x1.shape == x0.shape
    assert logp1.shape == (batch_shape[0],)
    assert_all_finite(x1, logp1)

    obs_dist = ssm.observation_dist(x1)
    y1 = obs_dist.sample()
    logp2 = obs_dist.log_prob(y1)
    assert y1.shape == (batch_shape[0], ssm.obs_dim)
    assert logp2.shape == (batch_shape[0],)
    assert_all_finite(y1, logp2)
