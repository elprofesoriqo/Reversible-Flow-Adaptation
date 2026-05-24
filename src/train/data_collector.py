# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
from src.models.teacher import TeacherPolicy
from src.env.quadruped_env import QuadrupedEnv

@jax.jit(static_argnames=('env', 'steps_per_env'))
def collect_trajectories(env: QuadrupedEnv, params, initial_state, key, steps_per_env):
    """Runs the Teacher policy to collect stochastic rollouts and compute values."""
    
    teacher = TeacherPolicy()
    
    def step_fn(state, key):
        key, action_key = jax.random.split(key)
        
        proprio = env.get_proprioceptive_obs(state)
        priv = env.get_privileged_obs(state)
        vision = env.get_vision_obs(state)
        
        # Forward pass: get distribution and value
        actor_mean, actor_log_std, critic_value = teacher.apply(params, proprio, priv)
        
        # Sample stochastic action for exploration using reparameterization trick
        actor_std = jnp.exp(actor_log_std)
        noise = jax.random.normal(action_key, actor_mean.shape)
        action = actor_mean + noise * actor_std
        
        # Calculate log probability of the sampled action
        var = jnp.square(actor_std)
        log_scale = actor_log_std + 0.5 * jnp.log(2.0 * jnp.pi)
        log_prob = -0.5 * jnp.sum(jnp.square(action - actor_mean) / var, axis=-1) - jnp.sum(log_scale, axis=-1)
        
        # Step env
        next_state, reward, done = env.step(state, action)
        
        # We must return the variables needed for GAE and PPO
        transition = (proprio, priv, vision, action, log_prob, critic_value, reward, done)
        return next_state, transition
        
    keys = jax.random.split(key, steps_per_env)
    final_state, transitions = jax.lax.scan(step_fn, initial_state, keys)
    
    return final_state, transitions
