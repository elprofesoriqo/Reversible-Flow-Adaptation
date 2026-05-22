# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
# pyrefly: ignore [missing-import]
import wandb
import time

from src.configs.h100_config import H100Config
from src.env.quadruped_env import QuadrupedEnv
from src.models.teacher import TeacherPolicy
from src.models.student import StudentPolicy
from src.train.ppo import ppo_update_step, compute_gae
from src.train.data_collector import collect_trajectories
from src.train.distillation import train_step as flow_matching_step
from src.train.replay_buffer import init_buffer, add_batch, sample_batch

def run_latency_benchmark(student_params, dummy_proprio, dummy_vision, dummy_noisy_act):
    """Benchmarks inference speed of Flow Matching vs DDPM."""
    student = StudentPolicy()
    
    @jax.jit
    def euler_integration_step(x_t, t):
        # Flow Matching Inference
        v_t, _ = student.apply(student_params, dummy_proprio, dummy_vision, x_t, t)
        return x_t + v_t
        
    @jax.jit
    def ddpm_20_step_inference(x_T):
        # 20-step DDPM iterative loop
        def body_fn(i, val):
            t = jnp.ones((dummy_noisy_act.shape[0], 1)) * (1.0 - i/20.0)
            noise_pred, _ = student.apply(student_params, dummy_proprio, dummy_vision, val, t)
            return val - 0.05 * noise_pred
        return jax.lax.fori_loop(0, 20, body_fn, x_T)
        
    # Compile and warmup
    x_T = dummy_noisy_act
    t_start = jnp.zeros((dummy_noisy_act.shape[0], 1))
    _ = euler_integration_step(x_T, t_start).block_until_ready()
    _ = ddpm_20_step_inference(x_T).block_until_ready()
    
    # Benchmark Flow Matching
    start_time = time.time()
    for _ in range(100):
        _ = euler_integration_step(x_T, t_start).block_until_ready()
    fm_time = (time.time() - start_time) / 100 * 1000 # ms per batch
    
    # Benchmark DDPM
    start_time = time.time()
    for _ in range(100):
        _ = ddpm_20_step_inference(x_T).block_until_ready()
    ddpm_time = (time.time() - start_time) / 100 * 1000 # ms per batch
    
    print(f"Flow Matching Latency: {fm_time:.2f} ms")
    print(f"DDPM Latency: {ddpm_time:.2f} ms")
    print(f"Speedup: {ddpm_time / fm_time:.2f}x")
    
    wandb.log({
        "benchmark/fm_latency_ms": fm_time,
        "benchmark/ddpm_latency_ms": ddpm_time,
        "benchmark/speedup_factor": ddpm_time / fm_time
    })

def main():
    config = H100Config()
    
    wandb.init(
        project=config.wandb_project,
        config=config.__dict__
    )
    
    key = jax.random.PRNGKey(42)
    
    # Environment & Buffer Initialization
    env = QuadrupedEnv(config)
    buffer_state = init_buffer(config)
    
    # Model Initialization
    teacher = TeacherPolicy()
    student = StudentPolicy()
    
    key, k1, k2 = jax.random.split(key, 3)
    dummy_proprio = jnp.zeros((config.batch_size, config.obs_dim_proprio))
    dummy_priv = jnp.zeros((config.batch_size, config.obs_dim_privileged))
    dummy_vision = jnp.zeros((config.batch_size, *config.vision_resolution))
    dummy_noisy_act = jnp.zeros((config.batch_size, config.chunk_size, config.action_dim))
    dummy_time = jnp.zeros((config.batch_size, 1))
    
    teacher_params = teacher.init(k1, dummy_proprio, dummy_priv)
    student_params = student.init(k2, dummy_proprio, dummy_vision, dummy_noisy_act, dummy_time)
    
    tx_teacher = optax.adam(3e-4)
    tx_student = optax.adamw(config.learning_rate)
    
    opt_state_teacher = tx_teacher.init(teacher_params)
    opt_state_student = tx_student.init(student_params)
    
    # Initial environment reset
    keys = jax.random.split(key, config.batch_size)
    batched_reset = jax.vmap(env.reset)
    current_states = batched_reset(keys)
    batched_collect = jax.vmap(collect_trajectories, in_axes=(None, None, 0, 0, None))
    batched_gae = jax.vmap(compute_gae)
        
    for epoch in range(1, 11):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # On-Policy Data Collection
        keys = jax.random.split(subkey1, config.batch_size)
        current_states, transitions = batched_collect(env, teacher_params, current_states, keys, config.chunk_size)
        proprio, priv, vision, action, log_prob, values, rewards, dones = transitions
        
        # Replay Buffer Formatting (Action Chunks)
        chunk_proprio = proprio[:, 0, :]
        chunk_priv = priv[:, 0, :]
        chunk_vision = vision[:, 0, :]
        chunk_action = action # Shape: (batch_size, chunk_size, action_dim)
        buffer_state = add_batch(buffer_state, (chunk_proprio, chunk_priv, chunk_vision, chunk_action))
        
        # Compute Generalized Advantage Estimation
        # Get next value for bootstrap
        _, _, next_value = jax.vmap(teacher.apply, in_axes=(None, 0, 0))(teacher_params, env.get_proprioceptive_obs(current_states), env.get_privileged_obs(current_states))
        
        advantages, returns = batched_gae(rewards, values, next_value, dones)
        
        # Reshape for PPO Batch
        # (batch_size, chunk_size, dim) -> (batch_size * chunk_size, dim)
        def flatten_batch(x):
            return x.reshape(-1, *x.shape[2:])
            
        ppo_batch = (
            flatten_batch(proprio), flatten_batch(priv), flatten_batch(action), 
            flatten_batch(log_prob), flatten_batch(advantages), flatten_batch(returns)
        )
        
        # PPO Teacher Update (On-Policy)
        teacher_params, opt_state_teacher, ppo_metrics = ppo_update_step(
            teacher_params, opt_state_teacher, ppo_batch, tx_teacher
        )
        
        # Flow Matching Distillation (Off-Policy from Replay Buffer)
        distill_batch = sample_batch(buffer_state, subkey2, config.batch_size)
        student_params, opt_state_student, fm_metrics = flow_matching_step(
            student_params, opt_state_student, distill_batch, subkey2, tx_student, config
        )
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            **ppo_metrics,
            **fm_metrics
        })
        
        print(f"Epoch {epoch} | Teacher Loss: {ppo_metrics['teacher_total_loss']:.4f} | Student FM: {fm_metrics['fm_loss']:.4f} | Student Aux: {fm_metrics['aux_loss']:.4f}")
        
    # Benchmark
    run_latency_benchmark(student_params, dummy_proprio, dummy_vision, dummy_noisy_act)
    
    wandb.finish()

if __name__ == "__main__":
    main()
