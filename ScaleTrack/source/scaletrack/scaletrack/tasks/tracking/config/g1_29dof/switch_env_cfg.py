from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from scaletrack.tasks.tracking.mdp import switch_rewards
from scaletrack.tasks.tracking.mdp.switch_commands import SwitchMotionCommandCfg
from scaletrack.tasks.tracking.tracking_env_cfg import CommandsCfg, RewardsCfg, TerminationsCfg

from .flat_env_cfg import G1BFMTrackingEnvCfg


@configclass
class SwitchCommandsCfg(CommandsCfg):
    motion = SwitchMotionCommandCfg(**{
        key: value for key, value in vars(CommandsCfg().motion).items() if key != "class_type"
    })


@configclass
class SwitchRewardsCfg(RewardsCfg):
    switch_sole_slip = RewardTermCfg(
        func=switch_rewards.switch_slip, weight=-1.25,
        params={"command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)},
    )
    switch_landing = RewardTermCfg(
        func=switch_rewards.switch_impact, weight=-0.4,
        params={"command_name": "motion", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    switch_pd_change = RewardTermCfg(
        func=switch_rewards.switch_torque_change, weight=-0.35, params={"command_name": "motion"},
    )
    switch_fall_penalty = RewardTermCfg(
        func=switch_rewards.switch_fall, weight=-20.0, params={"command_name": "motion"},
    )


@configclass
class SwitchTerminationsCfg(TerminationsCfg):
    switch_fall = TerminationTermCfg(
        func=switch_rewards.switch_fall, params={"command_name": "motion", "disable_flag": False},
    )


@configclass
class G1BFMSwitchTrackingEnvCfg(G1BFMTrackingEnvCfg):
    commands: SwitchCommandsCfg = SwitchCommandsCfg()
    rewards: SwitchRewardsCfg = SwitchRewardsCfg()
    terminations: SwitchTerminationsCfg = SwitchTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.debug_vis = False
        # Record all four physics substeps, including force spikes between actions.
        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        for target in ("height", "pos", "rot", "lin_vel", "ang_vel"):
            term = getattr(self.rewards, f"motion_body_{target}")
            term.func = switch_rewards.switch_tracking
            term.params = {**term.params, "target": target}
        self.terminations.body_pos.func = switch_rewards.switch_body_error
