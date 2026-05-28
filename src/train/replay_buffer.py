# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
from typing import NamedTuple, Tuple
# pyrefly: ignore [missing-import]
import flax.struct

class Transition(NamedTuple):
    proprio_obs: jax.Array
    privileged_obs: jax.Array
    vision_obs: jax.Array
    action_chunks: jax.Array


class ReplayBufferState(flax.struct.PyTreeNode):
    """ JAX  representation of the replay buffer state."""
    proprio_obs: jax.Array
    privileged_obs: jax.Array
    vision_obs: jax.Array
    action_chunks: jax.Array
    ptr: jax.Array
    size: jax.Array
    capacity: int = flax.struct.field(pytree_node=False)

def init_buffer(config) -> ReplayBufferState:
    """Initializes a zeroed-out buffer state in VRAM."""
    return ReplayBufferState(
        proprio_obs=jnp.zeros((config.buffer_capacity, config.obs_dim_proprio)),
        privileged_obs=jnp.zeros((config.buffer_capacity, config.obs_dim_privileged)),
        vision_obs=jnp.zeros((config.buffer_capacity, *config.vision_resolution)),
        action_chunks=jnp.zeros((config.buffer_capacity, config.chunk_size, config.action_dim)),
        ptr=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        capacity=config.buffer_capacity
    )

@jax.jit
def add_batch(buffer_state: ReplayBufferState, batch: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> ReplayBufferState:
    """Adds a batch of data using functional dynamic_update_slice to stay on GPU."""
    proprio, priv, vision, action_chunk = batch
    batch_size = proprio.shape[0]
    
    indices = (buffer_state.ptr + jnp.arange(batch_size)) % buffer_state.capacity
    
    new_proprio = buffer_state.proprio_obs.at[indices].set(proprio)
    new_priv = buffer_state.privileged_obs.at[indices].set(priv)
    new_vision = buffer_state.vision_obs.at[indices].set(vision)
    new_chunks = buffer_state.action_chunks.at[indices].set(action_chunk)
    
    new_ptr = (buffer_state.ptr + batch_size) % buffer_state.capacity
    new_size = jnp.minimum(buffer_state.size + batch_size, buffer_state.capacity)
    
    return ReplayBufferState(
        proprio_obs=new_proprio,
        privileged_obs=new_priv,
        vision_obs=new_vision,
        action_chunks=new_chunks,
        ptr=new_ptr,
        size=new_size,
        capacity=buffer_state.capacity
    )

@jax.jit(static_argnames=('batch_size',))
def sample_batch(buffer_state: ReplayBufferState, key: jax.Array, batch_size: int) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples a random off-policy batch from the populated buffer."""
    maxval = jnp.maximum(buffer_state.size, 1) 
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=maxval)
    
    sampled_proprio = buffer_state.proprio_obs[indices]
    sampled_priv = buffer_state.privileged_obs[indices]
    sampled_vision = buffer_state.vision_obs[indices]
    sampled_chunks = buffer_state.action_chunks[indices]
    
    return sampled_proprio, sampled_priv, sampled_vision, sampled_chunks
