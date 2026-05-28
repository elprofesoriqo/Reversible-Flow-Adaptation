# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import flax.linen as nn
from src.models.components import VisionEncoder, UNet1D, PhysicsHead
from typing import Tuple

class StudentPolicy(nn.Module):
    """Generative Policy with Action Chunking and Auxiliary Privileged Distillation."""
    obs_dim_privileged: int = 6
    action_dim: int = 12
    @nn.compact
    def __call__(self, proprio_obs: jax.Array, vision_obs: jax.Array, noisy_actions: jax.Array, time_steps: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        vision_features, cls_token = VisionEncoder()(vision_obs)
        
        # Auxiliary Head: Predict privileged state from vision (Legacy)
        priv_pred = nn.Dense(self.obs_dim_privileged)(vision_features)
        
        # New Physics Head: Predict mu, m, n_hat using LoRA
        physics_pred = PhysicsHead(out_dim=self.obs_dim_privileged)(cls_token)
        
        # Combine proprioception and vision features
        cond_features = jnp.concatenate([proprio_obs, vision_features], axis=-1)
        
        # Vector Field Prediction (Flow Matching)
        vector_field_pred = UNet1D(action_dim=self.action_dim)(noisy_actions, time_steps, cond_features)
        
        return vector_field_pred, priv_pred, physics_pred
