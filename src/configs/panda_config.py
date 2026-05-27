from dataclasses import dataclass

@dataclass(frozen=True)
class PandaConfig:
    num_envs: int = 8192
    batch_size: int = 64
    
    action_dim: int = 7          # 7 DoF Panda Arm
    obs_dim_proprio: int = 14    # Joint pos (7), vel (7)
    obs_dim_privileged: int = 2  # Target object mass (1), Object friction (1)
    vision_feat_dim: int = 256
    vision_resolution: tuple = (64, 64, 1) # Spatial depth map
    
    vit_patch_size: int = 8
    vit_num_heads: int = 4
    vit_num_layers: int = 2
    
    chunk_size: int = 16
    diffusion_steps: int = 20
    
    learning_rate: float = 3e-4
    aux_loss_weight: float = 0.1
    num_epochs: int = 600
    
    buffer_capacity: int = 100_000
    
    wandb_project: str = "agile-locomotion-flow"
    
    asset_path: str = "src/assets/mujoco_menagerie/franka_emika_panda/scene.xml"
