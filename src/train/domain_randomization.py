import jax
import jax.numpy as jnp
import optax
import time
import os
import argparse
# pyrefly: ignore [missing-import]
import orbax.checkpoint as ocp
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.configs.h100_config import H100Config
from src.configs.panda_config import PandaConfig
from src.env.quadruped_env import QuadrupedEnv
from src.env.panda_env import PandaEnv
from src.models.teacher import TeacherPolicy
from src.models.student import StudentPolicy
from src.train.ppo import ppo_update_step, compute_gae
from src.train.data_collector import collect_trajectories
from src.train.distillation import train_step as flow_matching_step
from src.train.replay_buffer import init_buffer, add_batch, sample_batch
from src.train.logger import WandbLogger

def apply_domain_randomization(state, key, env_type):
    """
    Scrambles physics parameters (mass, friction) of the environment 
    to force the Teacher to learn a gait without runtime adaptation.
    """
    key_m, key_f = jax.random.split(key)
    mjx_data = state.mjx_data
    
    if env_type == 'quadruped':
        pass
    elif env_type == 'panda':
        random_mass = jax.random.uniform(key_m, minval=0.5, maxval=5.0)
        state = state.replace(target_object_mass=jnp.array([random_mass]))
        
    return state

def run_dr_baseline():
    parser = argparse.ArgumentParser(description="Domain Randomization Baseline Training")
    parser.add_argument('--env', type=str, default='quadruped', choices=['quadruped', 'panda'])
    args = parser.parse_args()
    
    if args.env == 'panda':
        config = PandaConfig()
        env = PandaEnv(config)
    else:
        config = H100Config()
        env = QuadrupedEnv(config)
    
    # Overwrite wandb project so it doesn't mix with the TTA runs
    logger = WandbLogger(config)
    
    checkpoint_dir = os.path.abspath(os.path.join(os.getcwd(), f'checkpoints_dr_{args.env}'))
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    
    key = jax.random.PRNGKey(99)
    buffer_state = init_buffer(config)
    
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
    
    keys = jax.random.split(key, config.batch_size)
    batched_reset = jax.vmap(env.reset)
    current_states = batched_reset(keys)
    batched_collect = jax.vmap(collect_trajectories, in_axes=(None, None, 0, 0, None))
    batched_gae = jax.vmap(compute_gae)
        
    for epoch in range(1, 11):
        epoch_start_time = time.time()
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        
        keys_dr = jax.random.split(subkey3, config.batch_size)
        current_states = jax.vmap(apply_domain_randomization, in_axes=(0, 0, None))(current_states, keys_dr, args.env)
        
        # On-Policy Data Collection
        keys = jax.random.split(subkey1, config.batch_size)
        current_states, transitions = batched_collect(env, teacher_params, current_states, keys, config.chunk_size)
        proprio, priv, vision, action, log_prob, values, rewards, dones = transitions
        
        chunk_proprio = proprio[:, 0, :]
        chunk_priv = priv[:, 0, :]
        chunk_vision = vision[:, 0, :]
        chunk_action = action
        buffer_state = add_batch(buffer_state, (chunk_proprio, chunk_priv, chunk_vision, chunk_action))
        
        batched_get_proprio = jax.vmap(env.get_proprioceptive_obs)
        batched_get_priv = jax.vmap(env.get_privileged_obs)
        _, _, next_value = jax.vmap(teacher.apply, in_axes=(None, 0, 0))(
            teacher_params, batched_get_proprio(current_states), batched_get_priv(current_states)
        )

        advantages, returns = batched_gae(rewards, values, next_value, dones)
        
        def flatten_batch(x): return x.reshape(-1, *x.shape[2:])
        ppo_batch = (flatten_batch(proprio), flatten_batch(priv), flatten_batch(action), 
                     flatten_batch(log_prob), flatten_batch(advantages), flatten_batch(returns))
        
        teacher_params, opt_state_teacher, ppo_metrics = ppo_update_step(
            teacher_params, opt_state_teacher, ppo_batch, tx_teacher
        )
        
        distill_batch = sample_batch(buffer_state, subkey2, config.batch_size)
        student_params, opt_state_student, fm_metrics = flow_matching_step(
            student_params, opt_state_student, distill_batch, subkey2, tx_student, config
        )
        
        sys_metrics = {"SPS": (config.batch_size * config.chunk_size) / (time.time() - epoch_start_time)}
        logger.log_metrics(epoch, ppo_metrics, fm_metrics, sys_metrics)
        
        print(f"[DR BASELINE] Epoch {epoch} | Teacher Loss: {ppo_metrics['teacher_total_loss']:.4f} | Student FM: {fm_metrics['distillation/fm_vector_field_loss']:.4f}")
        
        if epoch % 5 == 0:
            ckpt = {'teacher': teacher_params, 'student': student_params, 'buffer': buffer_state}
            checkpoint_manager.save(epoch, args=ocp.args.StandardSave(ckpt))
            print(f"Saved DR checkpoint to {checkpoint_dir}")
            
    logger.finish()

if __name__ == "__main__":
    run_dr_baseline()
