from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from my_rsl_rl.networks import PULSEVAE


class VAEPolicy(nn.Module):
    """PULSE student: sampled posterior actions for training, posterior mean for play."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        inference_only: bool = False,
    ) -> None:
        super().__init__()
        self.obs_groups = {group: list(keys) for group, keys in obs_groups.items()}
        obs_dims = {key: obs[key].shape[-1] for keys in self.obs_groups.values() for key in keys}
        num_proprio_obs = sum(obs_dims[key] for key in self.obs_groups["policy"])
        num_target_obs = sum(obs_dims[key] for key in self.obs_groups["target"])
        self.policy_config = {
            "obs_groups": self.obs_groups,
            "obs_dims": obs_dims,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }
        self.model = PULSEVAE(
            num_proprio_obs, num_target_obs, num_actions, hidden_dim, latent_dim, inference_only=inference_only
        )

    def _get_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = torch.cat([obs[key] for key in self.obs_groups["policy"]], dim=-1)
        target = torch.cat([obs[key] for key in self.obs_groups["target"]], dim=-1)
        return proprio, target

    def forward(self, obs: TensorDict):
        proprio, target = self._get_obs(obs)
        return self.model(target, proprio)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        proprio, target = self._get_obs(obs)
        latent, _, _ = self.model.encode(target, proprio)
        return self.model.decode(latent, proprio)

    @torch.no_grad()
    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        proprio, target = self._get_obs(obs)
        mu, _ = self.model.encoder.encode(target, proprio)
        return self.model.decode(mu, proprio)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("model.prior_encoder.")}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str | torch.device = "cpu") -> VAEPolicy:
        config = checkpoint["policy_config"]
        obs = TensorDict(
            {key: torch.zeros(1, dim, device=device) for key, dim in config["obs_dims"].items()},
            batch_size=[1],
        )
        policy = cls(
            obs,
            config["obs_groups"],
            config["num_actions"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            inference_only=True,
        ).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return policy.eval()
