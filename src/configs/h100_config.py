"""
Hyperparameters tailored for NVIDIA H100 (80GB VRAM).
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class H100Config:
    num_envs: int = 8192
    batch_size: int = 256
    
    action_dim: int = 12         # 12 DoF Quadruped
    obs_dim_proprio: int = 27    # Joint pos (12), vel (12), projected gravity (3)
    obs_dim_privileged: int = 6  # Root linear vel (3), angular vel (3)
    vision_feat_dim: int = 256
    vision_resolution: tuple = (64, 64, 1) # Spatial depth map dimensions (H, W, C)
    
    vit_patch_size: int = 8
    vit_num_heads: int = 4
    vit_num_layers: int = 2
    
    chunk_size: int = 16         # Action chunking sequence length
    diffusion_steps: int = 20
    
    learning_rate: float = 3e-4
    aux_loss_weight: float = 0.1 # Weight for privileged state prediction
    num_epochs: int = 600
    
    buffer_capacity: int = 100_000
    
    wandb_project: str = "agile-locomotion-flow"
    
    asset_path: str = "src/assets/mujoco_menagerie/unitree_go2/scene.xml"
