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
    experiment_name: str = "g1_bfm_distillation"
    run_name: str = "debug"
    wandb_project: str = "ScaleBFM"
    neptune_project: str = "ScaleBFM"
    empirical_normalization: bool = False
    obs_groups: dict = {"policy": ["student_proprio"], "target": ["student_target"]}
    observation: DistillationObservationCfg = DistillationObservationCfg()
    teacher: DistillationTeacherCfg = DistillationTeacherCfg()
    policy: dict = {"class_name": "VAEPolicy", "hidden_dim": 256, "latent_dim": 32}
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
