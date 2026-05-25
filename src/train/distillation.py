# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
import os
import numpy as np
from src.models.teacher import TeacherPolicy
from src.models.student import StudentPolicy
from src.models.components import InvertibleProjectionMapping

def loss_fn(student_params, teacher_action_chunks, proprio_obs, privileged_obs, vision_obs, key, config):
    """Computes Flow Matching Vector Field Loss, Auxiliary Distillation Loss, and Physics Head Loss."""
    batch_size = teacher_action_chunks.shape[0]
    key_t, key_noise = jax.random.split(key)
    
    # Sample continuous timestep t in [0, 1]
    t = jax.random.uniform(key_t, (batch_size, 1))
    
    # x_1 is the target (teacher action chunk)
    x_1 = teacher_action_chunks
    # x_0 is pure noise
    x_0 = jax.random.normal(key_noise, x_1.shape)
    
    # Optimal Transport Interpolation (Straight line)
    t_expanded = t[:, :, None] # Match broadcast dimensions
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    
    # The target vector field is the derivative of the path: v_t = x_1 - x_0
    target_vector_field = x_1 - x_0
    
    # Predict Vector Field and Privileged State with Student Policy
    student = StudentPolicy()
    vector_field_pred, priv_pred, physics_pred = student.apply(
        student_params, proprio_obs, vision_obs, x_t, t
    )
    
    # Flow Matching MSE Loss
    fm_loss = jnp.mean((vector_field_pred - target_vector_field) ** 2)
    
    # Auxiliary Distillation MSE Loss (Predicting Teacher's Privileged Data)
    aux_loss = jnp.mean((priv_pred - privileged_obs) ** 2)
    
    # Physics Head MSE Loss (Predicting physical parameters: mu, m, n_hat)
    physics_target = privileged_obs[..., :5]
    aux_physics_mse = jnp.mean((physics_pred - physics_target) ** 2)
    
    # Total Loss
    total_loss = fm_loss + config.aux_loss_weight * aux_loss + config.aux_loss_weight * aux_physics_mse
    
    metrics = {
        "distillation/fm_vector_field_loss": fm_loss,
        "distillation/aux_loss": aux_loss,
        "distillation/aux_physics_mse": aux_physics_mse,
        "distillation/action_divergence": fm_loss, # Proxy metric during pretraining
        "student_total_loss": total_loss
    }
    
    return total_loss, metrics

@jax.jit(static_argnames=('config', 'tx'))
def train_step(student_params, opt_state, batch, key, tx, config):
    """
    HLO-compiled training step for Flow Matching operating on off-policy ReplayBuffer batches.
    """
    proprio_obs, privileged_obs, vision_obs, action_chunks = batch
    
    # Compute Gradients
    student = StudentPolicy()
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        student_params, action_chunks, 
        proprio_obs, privileged_obs, vision_obs, key, config
    )
    
    # Update Student
    updates, opt_state = tx.update(grads, opt_state, student_params)
    new_student_params = optax.apply_updates(student_params, updates)
    
    return new_student_params, opt_state, metrics

def train_invertible_projection_offline(config, tx, num_epochs=1000):
    """
    Offline training loop to pretrain the Invertible Projection Mapping 
    on the harvested PD torque dataset via Maximum Likelihood Estimation.
    """
    dataset_path = 'data/pd_torques_dataset.npy'
        
    dataset = np.load(dataset_path)
    
    model = InvertibleProjectionMapping(hidden_dim=64)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, config.action_dim))
    params = model.init(key, dummy_input)
    
    opt_state = tx.init(params)
    
    def nll_loss_fn(p, batch):
        # Maximize the likelihood of the PD torques under the normalizing flow: log p(x) = log p(z) + log |det J|
        z, log_det = model.apply(p, batch, return_log_det=True)
        
        # Base distribution is standard normal N(0, I)
        log_pz = -0.5 * jnp.sum(jnp.square(z), axis=-1) - 0.5 * z.shape[-1] * jnp.log(2 * jnp.pi)
        
        log_px = log_pz + log_det
        return -jnp.mean(log_px) # Minimize NLL
        
    @jax.jit
    def proj_step_fn(p, o_state, batch):
        loss, grads = jax.value_and_grad(nll_loss_fn)(p, batch)
        updates, o_state = tx.update(grads, o_state, p)
        new_params = optax.apply_updates(p, updates)
        return new_params, o_state, loss
        
    batch_size = min(config.batch_size, dataset.shape[0])
    num_samples = dataset.shape[0]
    
    for epoch in range(num_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch = dataset[i:i+batch_size]
            if batch.shape[0] < batch_size:
                continue
                
            params, opt_state, loss = proj_step_fn(params, opt_state, jnp.array(batch))
            epoch_loss += loss
            batches += 1
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} - NLL Loss: {epoch_loss / max(1, batches):.4f}")
            
    return params
