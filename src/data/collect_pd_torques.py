import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

from src.configs.h100_config import H100Config
from src.env.quadruped_env import QuadrupedEnv

def collect_offline_pd_data():
    """
    Offline data collection script that induces OOD events 
    (dropping mass, changing friction) and records the PD reflex's torque responses.
    This dataset is strictly used to pretrain the Invertible Projection Mapping.
    """
    config = H100Config()
    env = QuadrupedEnv(config)
    
    key = jax.random.PRNGKey(0)
    
    # collect 10,000 steps of OOD responses
    num_steps = 10000
    
    @jax.jit
    def step_fn(state, key):
        # Generate random actions
        action = jax.random.normal(key, (config.action_dim,))
        
        # induce OOD by injecting a massive force disturbance into the physics engine to trigger a torque mismatch.
        mjx_data = state.mjx_data
        disturbance = jax.random.normal(key, (6,)) * 100.0
        qfrc_applied = jnp.zeros_like(mjx_data.qfrc_applied)
        qfrc_applied = qfrc_applied.at[..., :6].set(disturbance)
        
        mjx_data = mjx_data.replace(qfrc_applied=qfrc_applied)
        state = state.replace(mjx_data=mjx_data)
        
        next_state, reward, done, info = env.step(state, action)
        return next_state, info['a_corr']
        
    state = env.reset(key)
    
    a_corr_dataset = []
    
    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        state, a_corr = step_fn(state, subkey)
        a_corr_dataset.append(np.array(a_corr))
        
        if (i + 1) % 2000 == 0:
            print(f"Collected {i + 1}/{num_steps} samples...")
            
    a_corr_dataset = np.stack(a_corr_dataset)
    
    os.makedirs('data', exist_ok=True)
    np.save('data/pd_torques_dataset.npy', a_corr_dataset)
    print(f"Collection complete. Saved shape: {a_corr_dataset.shape} to data/pd_torques_dataset.npy")

if __name__ == "__main__":
    collect_offline_pd_data()
