from isaaclab.utils import configclass

from scaletrack.robots.g1_29dof import G1_29DOF_ACTION_SCALE, G1_29DOF_CYLINDER_CFG

from scaletrack.tasks.tracking.tracking_env_cfg import (
    BFMTrackingEnvCfg,
)

@configclass
class G1BFMTrackingEnvCfg(BFMTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_29DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_29DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.commands.motion.mode_candidates = {
            "Pelvis-1": ["pelvis"],
            "UMI-2": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "VR-3": ["pelvis","left_wrist_yaw_link","right_wrist_yaw_link"],
            "UMI-4": ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
            "VR-5": ["pelvis","left_wrist_yaw_link","right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
            "UpperBody-6": [
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link"
            ],
            "UpperBody-Mobile-7": [
                "pelvis",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link"
            ],
            "WholeBody-14":  [
                "pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ]
        }


@configclass
class G1BFMDistillationEnvCfg(G1BFMTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.mode_candidates = {"WholeBody-14": self.commands.motion.body_names.copy()}
