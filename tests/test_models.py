import torch

from hai_zd.models import ConvVAE, DeepSVDD, svdd_sensor_scores, vae_sensor_scores
from hai_zd.train import _elbo


def test_deep_svdd_is_bias_free_and_scores_each_sensor():
    model = DeepSVDD(width=4, latent_dim=3)
    assert all(module.bias is None for module in model.modules() if hasattr(module, "bias"))
    model.center.fill_(0.1)
    scores = svdd_sensor_scores(model, torch.randn(2, 5, 60))
    assert scores.shape == (2, 5)
    assert torch.all(scores >= 0)


def test_vae_elbo_sums_reconstruction_and_kl_per_sample():
    x = torch.ones(1, 1, 4)
    reconstruction = torch.zeros_like(x)
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)
    assert _elbo(x, reconstruction, mu, logvar).item() == 2.0
    assert _elbo(x, x, mu, logvar).item() == 0.0


def test_vae_scoring_is_deterministic_and_uses_full_window():
    model = ConvVAE(width=4, latent_dim=3, window=60).eval()
    x = torch.randn(2, 5, 60)
    first = vae_sensor_scores(model, x)
    second = vae_sensor_scores(model, x)
    assert first.shape == (2, 5)
    assert torch.equal(first, second)
    assert torch.all(first >= 0)
