"""Student observations on top of the unchanged RSL-RL teacher environment."""

import torch
from tensordict import TensorDict

from my_rsl_rl.env import VecEnv


class DistillationVecEnvWrapper(VecEnv):
    """Append the current-frame VAE inputs to an RslRlVecEnvWrapper."""

    def __init__(self, env, cfg: dict | None = None):
        self.env = env
        self.observation_config = {"gyro_scale": 0.05, "joint_velocity_scale": 0.05, **(cfg or {})}
        self.motion = self.unwrapped.command_manager.get_term("motion")
        self.robot = self.motion.robot
        self.action_term = self.unwrapped.action_manager.get_term("joint_pos")
        if self.num_actions != 29 or list(self.action_term._joint_names) != list(self.robot.joint_names):
            raise ValueError("Distillation requires all 29 joints in native articulation order.")
        manager = self.unwrapped.observation_manager
        terms = manager._group_obs_term_cfgs["policy"]
        expected_names = ["projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"]
        if manager.active_terms["policy"] != expected_names or any(
            not torch.as_tensor(term.scale if term.scale is not None else 1.0).eq(scale).all().item()
            for term, scale in zip(terms, (1.0, 1.0, 1.0, 0.05))
        ):
            raise ValueError("Distillation requires the standard teacher policy term order and scales.")
        self.last_motor_targets = self.robot.data.joint_pos.clone()

        # Startup calibration randomization retains this unbatched nominal vector.
        nominal_offset = getattr(
            self.robot.data, "default_joint_pos_nominal", self.robot.data.default_joint_pos[0]
        )
        scale = torch.as_tensor(self.action_term._scale).broadcast_to((self.num_envs, self.num_actions))[0]
        self.action_config = {
            "joint_names": list(self.action_term._joint_names),
            "action_scale": scale.detach().cpu().tolist(),
            "nominal_action_offset": nominal_offset.detach().cpu().tolist(),
            "control_dt": float(self.unwrapped.step_dt),
        }

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def validate_action_config(self, config: dict) -> None:
        """Reject playback in an environment with a different action contract."""
        if config != self.action_config:
            raise ValueError("Checkpoint action configuration does not match the environment.")

    def get_observations(self) -> TensorDict:
        return self._student_observations(self.env.get_observations())

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        self.last_motor_targets.copy_(self.robot.data.joint_pos)
        return self._student_observations(obs), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        self.last_motor_targets.copy_(self.action_term.processed_actions)
        # Isaac Lab returns the new episode's observation for automatically reset rows.
        reset_rows = dones.bool()
        self.last_motor_targets[reset_rows] = self.robot.data.joint_pos[reset_rows]
        return self._student_observations(obs), rewards, dones, extras

    def _student_observations(self, obs: TensorDict) -> TensorDict:
        # The teacher frame is [gravity, angular velocity, relative q, 0.05 * relative dq].
        gravity, gyro, joint_pos, joint_vel = obs["policy"][:, -1].split([3, 3, 29, 29], dim=-1)
        velocity_scale = self.observation_config["joint_velocity_scale"]
        joint_vel = joint_vel * (velocity_scale / 0.05)
        student_proprio = torch.cat(
            [gravity, gyro * self.observation_config["gyro_scale"], joint_pos, joint_vel, self.last_motor_targets],
            dim=-1,
        )
        # Recover the same noisy joint state, including per-environment calibration offsets.
        position_error = self.motion.joint_pos - (joint_pos + self.robot.data.default_joint_pos)
        velocity_error = (
            self.motion.joint_vel - self.robot.data.default_joint_vel
        ) * velocity_scale - joint_vel
        result = obs.copy()
        result["student_proprio"] = student_proprio
        result["student_target"] = torch.cat([position_error, velocity_error], dim=-1)
        return result
