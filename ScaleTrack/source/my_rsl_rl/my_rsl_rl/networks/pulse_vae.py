from __future__ import annotations

import torch
from torch import nn

from .humanoid_transformer import HumanoidTransformer


class _Layer(nn.Module):
    """Keep the original PULSE parameter names for weight comparisons."""

    def __init__(self, input_dim: int, output_dim: int, activate: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ELU() if activate else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns transformed features with shape (..., output_dim).
        return self.net(x)


def _sample(mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    # Returns a reparameterized Gaussian sample with the same shape as mu.
    if noise is None:
        noise = torch.randn_like(mu)
    return mu + noise * torch.exp(0.5 * logvar)


class ConditionalEncoder(nn.Module):
    def __init__(self, condition_dim: int, data_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = _Layer(data_dim + condition_dim, hidden_dim)
        self.fc2 = _Layer(data_dim + hidden_dim, hidden_dim)
        self.mu = nn.Linear(data_dim + hidden_dim, latent_dim)
        self.logvar = nn.Linear(data_dim + hidden_dim, latent_dim)

    def encode(self, data: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns posterior (mean, log_variance), each with shape (..., latent_dim).
        h1 = self.fc1(torch.cat((data, condition), dim=-1))
        h2 = self.fc2(torch.cat((data, h1), dim=-1))
        features = torch.cat((data, h2), dim=-1)
        return self.mu(features), self.logvar(features)

    def forward(self, data: torch.Tensor, condition: torch.Tensor, noise: torch.Tensor | None = None):
        # Returns posterior (sample, mean, log_variance), each with shape (..., latent_dim).
        mu, logvar = self.encode(data, condition)
        return _sample(mu, logvar, noise), mu, logvar


class TransformerPosterior(nn.Module):
    """Encode policy/action history and future task tokens into a Gaussian latent."""

    def __init__(
        self,
        prop_obs_dim: int,
        action_dim: int,
        latent_dim: int,
        embedding_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.prop_obs_dim = prop_obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.transformer = HumanoidTransformer(
            prop_obs_dim=prop_obs_dim,
            action_dim=action_dim,
            output_dim=2 * latent_dim,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
        )
        self.transformer.init_weights()
        # Start near N(0, I), while allowing gradients into the attention trunk immediately.
        nn.init.normal_(self.transformer.projection_head.weight, std=0.01 * embedding_dim ** -0.5)

    def encode(
        self,
        prop_obs: torch.Tensor,
        action_obs: torch.Tensor,
        task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B, latent_dim] statistics from the HumanoidTransformer inputs."""
        statistics = self.transformer(prop_obs, action_obs, task_tokens)
        return statistics.chunk(2, dim=-1)

    def forward(
        self,
        prop_obs: torch.Tensor,
        action_obs: torch.Tensor,
        task_tokens: torch.Tensor,
        noise: torch.Tensor | None = None,
    ):
        mu, logvar = self.encode(prop_obs, action_obs, task_tokens)
        return _sample(mu, logvar, noise), mu, logvar


class ConditionalPrior(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = _Layer(condition_dim, hidden_dim)
        self.fc2 = _Layer(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns prior (mean, log_variance), each with shape (..., latent_dim).
        features = self.fc2(self.fc1(condition))
        return self.mu(features), self.logvar(features)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor | None = None):
        # Returns prior (sample, mean, log_variance), each with shape (..., latent_dim).
        mu, logvar = self.encode(condition)
        return _sample(mu, logvar, noise), mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(self, condition_dim: int, data_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = _Layer(latent_dim + condition_dim, hidden_dim)
        self.fc2 = _Layer(latent_dim + hidden_dim, hidden_dim)
        self.out = _Layer(latent_dim + hidden_dim, data_dim, activate=False)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Returns reconstructed data with shape (..., data_dim).
        h1 = self.fc1(torch.cat((latent, condition), dim=-1))
        h2 = self.fc2(torch.cat((latent, h1), dim=-1))
        return self.out(torch.cat((latent, h2), dim=-1))
