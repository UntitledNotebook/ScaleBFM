from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


@configclass
class DistillationObservationCfg:
    gyro_scale: float = 0.05
    joint_velocity_scale: float = 0.05


@configclass
class DistillationTeacherCfg:
    checkpoint: str = ""
    policy: dict = {
        "class_name": "ActorCriticHumanoidTransformer",
        "embedding_dim": 384,
        "num_heads": 6,
        "ff_dim": 384,
        "num_layers": 6,
        "use_transformer_critic": True,
        "task_embedder_hidden_dims": [],
        "activation": "elu",
        "init_noise_std": 0.8,
        "state_dependent_std": False,
    }


@configclass
class G1DistillationRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "DistillationRunner"
    task_name: str = "G1-BFM-Distillation-Tracking"
    num_steps_per_env: int = 1
    max_iterations: int = 400000
    save_interval: int = 10000
    eval_during_training: bool = True
    eval_interval: int = 10000
    eval_max_steps: int = 1000
    eval_metric_keys: list[str] = [
        "error_anchor_height", "error_anchor_rot", "error_anchor_pos",
        "error_anchor_lin_vel", "error_anchor_ang_vel", "error_body_pos_g",
        "error_body_pos", "error_body_pos_relative", "error_body_rot", "error_body_rot_relative",
        "error_joint_pos", "error_joint_vel",
    ]
    success_metric_dict: dict = {"error_body_pos_g": 0.5}
    command_name: str = "motion"
    experiment_name: str = "g1_bfm_distillation"
    run_name: str = "debug"
    wandb_project: str = "ScaleBFM"
    neptune_project: str = "ScaleBFM"
    empirical_normalization: bool = False
    obs_groups: dict = {
        "policy": ["student_proprio"],
        "posterior_policy": ["policy"],
        "posterior_task": ["policy_task"],
        "posterior_action": ["action"],
    }
    observation: DistillationObservationCfg = DistillationObservationCfg()
    teacher: DistillationTeacherCfg = DistillationTeacherCfg()
    policy: dict = {
        "class_name": "VAEPolicy",
        "hidden_dim": 256,
        "latent_dim": 32,
        "posterior_type": "transformer",
        "posterior_cfg": {
            "embedding_dim": 256,
            "num_heads": 4,
            "ff_dim": 256,
            "num_layers": 4,
        },
    }
    algorithm: dict = {
        "class_name": "Distillation",
        "num_learning_epochs": 5,
        "learning_rate": 8.0e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "kl_weight": 0.01,
        "temporal_weight": 0.0,
        "schedule": "cosine_annealing",
        "schedule_iterations": 400000,
        "min_learning_rate": 1.0e-6,
    }
