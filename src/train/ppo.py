# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
from src.models.teacher import TeacherPolicy

def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """Computes Generalized Advantage Estimation."""
    # Append next_value for bootstrap
    all_values = jnp.concatenate([values, next_value[None]], axis=0)
    
    # Calculate TD errors
    deltas = rewards + gamma * all_values[1:] * (1.0 - dones) - all_values[:-1]
    
    def scan_fn(gae_t_plus_1, transition):
        delta_t, done_t = transition
        gae_t = delta_t + gamma * gae_lambda * (1.0 - done_t) * gae_t_plus_1
        return gae_t, gae_t
        
    _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[0]), (deltas, dones), reverse=True)
    returns = advantages + values
    return advantages, returns

def ppo_loss_fn(params, apply_fn, obs_proprio, obs_priv, actions, old_log_probs, advantages, returns, clip_ratio=0.2):
    """PPO clipped surrogate objective with true Gaussian continuous control math."""
    
    # Forward pass: get parameterized distribution and value
    teacher = TeacherPolicy()
    actor_mean, actor_log_std, critic_value = teacher.apply(params, obs_proprio, obs_priv)
    actor_std = jnp.exp(actor_log_std)
    
    # Calculate new log probabilities of the actions taken during rollout
    # log_prob = -0.5 * (((x - mu) / sigma)^2 + 2*log(sigma) + log(2*pi))
    var = jnp.square(actor_std)
    log_scale = actor_log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    new_log_probs = -0.5 * jnp.sum(jnp.square(actions - actor_mean) / var, axis=-1) - jnp.sum(log_scale, axis=-1)
    
    # Policy ratio and surrogate objective
    ratio = jnp.exp(new_log_probs - old_log_probs)
    
    # Policy loss (clipped)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
    
    # True Value loss (MSE between Critic prediction and GAE returns)
    v_loss = jnp.mean(jnp.square(critic_value - returns))
    
    # True Entropy (for Gaussian: 0.5 + 0.5 * log(2*pi) + log_std)
    entropy = jnp.mean(jnp.sum(0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + actor_log_std, axis=-1))
    
    # Total loss (standard PPO coefficients)
    total_loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
    
    metrics = {
        "ppo_pg_loss": pg_loss,
        "ppo_v_loss": v_loss,
        "ppo_entropy": entropy,
        "teacher_total_loss": total_loss
    }
    
    return total_loss, metrics

@jax.jit(static_argnames=('tx',))
def ppo_update_step(params, opt_state, batch, tx):
    """Jitted PPO update over a batch of true on-policy transitions."""
    obs_proprio, obs_priv, actions, old_log_probs, advantages, returns = batch
    
    teacher = TeacherPolicy()
    
    (loss, metrics), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
        params, obs_proprio, obs_priv, actions, old_log_probs, advantages, returns
    )
    
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, opt_state, metrics
