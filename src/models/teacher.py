# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import flax.linen as nn
from typing import Tuple

class TeacherPolicy(nn.Module):
    """Rigorous Continuous PPO Actor-Critic Architecture."""
    action_dim: int = 12
    
    @nn.compact
    def __call__(self, proprio_obs: jax.Array, privileged_obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Shared features (optional, but standard)
        x = jnp.concatenate([proprio_obs, privileged_obs], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        # Actor Head (Mean)
        actor_mean = nn.Dense(self.action_dim)(x)
        
        # Actor Head (Log Std) - State independent standard deviation is common in PPO
        actor_log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        
        # Critic Head (Value)
        critic_value = nn.Dense(1)(x)
        critic_value = jnp.squeeze(critic_value, axis=-1)
        
        return actor_mean, actor_log_std, critic_value
