from __future__ import annotations

import torch
from torch import nn


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


class PULSEVAE(nn.Module):
    """Conditional VAE with a learned proprioceptive prior and one decoder."""

    def __init__(
        self,
        condition_dim: int,
        encoder_additional_input_dim: int,
        data_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.prior_encoder = ConditionalPrior(condition_dim, hidden_dim, latent_dim)
        self.encoder = ConditionalEncoder(condition_dim, encoder_additional_input_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(condition_dim, data_dim, hidden_dim, latent_dim)

    def encode(self, data: torch.Tensor, condition: torch.Tensor, noise: torch.Tensor | None = None):
        # Returns posterior (sample, mean, log_variance), each with shape (..., latent_dim).
        return self.encoder(data, condition, noise)

    def prior_encode(self, condition: torch.Tensor, noise: torch.Tensor | None = None):
        # Returns prior (sample, mean, log_variance), each with shape (..., latent_dim).
        return self.prior_encoder(condition, noise)

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Returns reconstructed data with shape (..., data_dim).
        return self.decoder(latent, condition)

    def forward(self, encoder_input: torch.Tensor, condition: torch.Tensor):
        # Returns reconstruction (..., data_dim), posterior (mu_q, logvar_q), and prior (mu_p, logvar_p), each statistic (..., latent_dim).
        latent, mu_q, logvar_q = self.encode(encoder_input, condition)
        # Retain the source model's posterior-then-prior sampling order.
        _, mu_p, logvar_p = self.prior_encode(condition)
        return self.decode(latent, condition), (mu_q, logvar_q), (mu_p, logvar_p)
