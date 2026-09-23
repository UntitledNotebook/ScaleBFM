from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch
from tensordict import TensorDict
from torch import nn

from my_rsl_rl.networks import ConditionalEncoder, ConditionalPrior, ConditionalDecoder, TaskEmbedder, TransformerPosterior


class VAEPolicy(nn.Module):
    """PULSE student: sampled posterior actions for training, posterior mean for play."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        posterior_type: Literal["mlp", "transformer"] = "mlp",
        posterior_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.posterior_type = posterior_type
        self.posterior_cfg = deepcopy(posterior_cfg or {})
        self.obs_groups = {group: list(keys) for group, keys in obs_groups.items()}
        num_proprio_obs = self._validate_group(obs, "policy", ndim=2)
        posterior_encoder = None
        if posterior_type == "mlp":
            if self.posterior_cfg:
                raise ValueError("posterior_cfg must be empty for the MLP posterior.")
            num_target_obs = self._validate_group(obs, "target", ndim=2)
        else:
            prop_dim = self._validate_group(obs, "posterior_policy", ndim=3)
            task_dim = self._validate_group(obs, "posterior_task", ndim=3)
            action_dim = self._validate_group(obs, "posterior_action", ndim=3)
            prop_history = obs[self.obs_groups["posterior_policy"][0]].shape[1]
            action_history = obs[self.obs_groups["posterior_action"][0]].shape[1]
            if prop_history != action_history:
                raise ValueError("Posterior policy and action histories must have the same length.")
            transformer_cfg = dict(self.posterior_cfg)
            task_embedder_hidden_dims = transformer_cfg.pop("task_embedder_hidden_dims", None)
            posterior_encoder = TransformerPosterior(
                prop_obs_dim=prop_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                **transformer_cfg,
            )
            self.posterior_task_embedder = TaskEmbedder(
                task_obs_dim=task_dim,
                embedding_dim=posterior_encoder.embedding_dim,
                hidden_dims=task_embedder_hidden_dims,
            )
            self.posterior_task_embedder.init_weights()
        obs_dims = {key: obs[key].shape[-1] for keys in self.obs_groups.values() for key in keys}
        self.policy_config = {
            "obs_groups": self.obs_groups,
            "obs_dims": obs_dims,
            "obs_shapes": {key: list(obs[key].shape[1:]) for key in obs_dims},
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "posterior_type": posterior_type,
            "posterior_cfg": deepcopy(self.posterior_cfg),
        }
        self.prior = ConditionalPrior(num_proprio_obs, hidden_dim, latent_dim)
        self.posterior = (
            ConditionalEncoder(num_proprio_obs, num_target_obs, hidden_dim, latent_dim)
            if posterior_encoder is None
            else posterior_encoder
        )
        self.decoder = ConditionalDecoder(num_proprio_obs, num_actions, hidden_dim, latent_dim)

    def _validate_group(self, obs: TensorDict, group: str, ndim: int) -> int:
        keys = self.obs_groups.get(group)
        if not keys:
            raise ValueError(f"The {self.posterior_type} posterior requires a nonempty '{group}' observation group.")
        shapes = [obs[key].shape for key in keys]
        if any(len(shape) != ndim or any(dim <= 0 for dim in shape[1:]) for shape in shapes):
            raise ValueError(f"Observation group '{group}' requires rank-{ndim} tensors with nonempty features.")
        if any(shape[:-1] != shapes[0][:-1] for shape in shapes):
            raise ValueError(f"Observations in group '{group}' must have matching batch and sequence dimensions.")
        return sum(shape[-1] for shape in shapes)

    def _get_group(self, obs: TensorDict, group: str) -> torch.Tensor:
        return torch.cat([obs[key] for key in self.obs_groups[group]], dim=-1)

    def forward(self, obs: TensorDict):
        proprio = self._get_group(obs, "policy")
        if self.posterior_type == "mlp":
            latent, mu_q, logvar_q = self.posterior(self._get_group(obs, "target"), proprio)
        else:
            latent, mu_q, logvar_q = self.posterior(
                self._get_group(obs, "posterior_policy"),
                self._get_group(obs, "posterior_action"),
                self.posterior_task_embedder(self._get_group(obs, "posterior_task")),
            )
        # Retain the source model's posterior-then-prior sampling order.
        _, mu_p, logvar_p = self.prior(proprio)
        return self.decoder(latent, proprio), (mu_q, logvar_q), (mu_p, logvar_p)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        proprio = self._get_group(obs, "policy")
        if self.posterior_type == "mlp":
            latent, _, _ = self.posterior(self._get_group(obs, "target"), proprio)
        else:
            latent, _, _ = self.posterior(
                self._get_group(obs, "posterior_policy"),
                self._get_group(obs, "posterior_action"),
                self.posterior_task_embedder(self._get_group(obs, "posterior_task")),
            )
        return self.decoder(latent, proprio)

    @torch.no_grad()
    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        proprio = self._get_group(obs, "policy")
        if self.posterior_type == "mlp":
            mu, _ = self.posterior.encode(self._get_group(obs, "target"), proprio)
        else:
            mu, _ = self.posterior.encode(
                self._get_group(obs, "posterior_policy"),
                self._get_group(obs, "posterior_action"),
                self.posterior_task_embedder(self._get_group(obs, "posterior_task")),
            )
        return self.decoder(mu, proprio)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str | torch.device = "cpu") -> VAEPolicy:
        """Restore the posterior encoder, decoder, and prior."""
        config = checkpoint["policy_config"]
        posterior_type = config.get("posterior_type", "mlp")
        if posterior_type == "transformer" and "obs_shapes" not in config:
            raise ValueError("Transformer checkpoints must include full observation shapes.")
        obs_shapes = config.get("obs_shapes")
        if obs_shapes is None:
            # Original MLP checkpoints stored only flat feature widths.
            obs_shapes = {key: [dim] for key, dim in config["obs_dims"].items()}
        obs = TensorDict(
            {key: torch.zeros(1, *shape, device=device) for key, shape in obs_shapes.items()},
            batch_size=[1],
        )
        policy = cls(
            obs,
            config["obs_groups"],
            config["num_actions"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            posterior_type=posterior_type,
            posterior_cfg=config.get("posterior_cfg"),
        ).to(device)
        state_dict = checkpoint["model_state_dict"]
        policy.load_state_dict(state_dict, strict=True)
        return policy.eval()
