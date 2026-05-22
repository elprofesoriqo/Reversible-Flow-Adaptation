# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import optax
from src.models.teacher import TeacherPolicy
from src.models.student import StudentPolicy

def loss_fn(student_params, student_apply_fn, teacher_action_chunks, proprio_obs, privileged_obs, vision_obs, key, config):
    """Computes Flow Matching Vector Field Loss and Auxiliary Distillation Loss."""
    batch_size = teacher_action_chunks.shape[0]
    key_t, key_noise = jax.random.split(key)
    
    # Sample continuous timestep t in [0, 1]
    t = jax.random.uniform(key_t, (batch_size, 1))
    
    # x_1 is the target (teacher action chunk)
    x_1 = teacher_action_chunks
    # x_0 is pure noise
    x_0 = jax.random.normal(key_noise, x_1.shape)
    
    # Optimal Transport Interpolation (Straight line)
    # x_t = t * x_1 + (1 - t) * x_0
    t_expanded = t[:, :, None] # Match broadcast dimensions
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    
    # The target vector field is the derivative of the path: v_t = x_1 - x_0
    target_vector_field = x_1 - x_0
    
    # Predict Vector Field and Privileged State with Student Policy
    vector_field_pred, priv_pred = student_apply_fn(
        student_params, proprio_obs, vision_obs, x_t, t
    )
    
    # Flow Matching MSE Loss
    fm_loss = jnp.mean((vector_field_pred - target_vector_field) ** 2)
    
    # Auxiliary Distillation MSE Loss (Predicting Teacher's Privileged Data)
    aux_loss = jnp.mean((priv_pred - privileged_obs) ** 2)
    
    # Total Loss
    total_loss = fm_loss + config.aux_loss_weight * aux_loss
    
    metrics = {
        "fm_loss": fm_loss,
        "aux_loss": aux_loss,
        "student_total_loss": total_loss
    }
    
    return total_loss, metrics

@jax.jit
def train_step(student_params, opt_state, batch, key, tx, config):
    """
    HLO-compiled training step for Flow Matching operating on off-policy ReplayBuffer batches.
    """
    proprio_obs, privileged_obs, vision_obs, action_chunks = batch
    
    # Compute Gradients
    student = StudentPolicy()
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        student_params, student.apply, action_chunks, 
        proprio_obs, privileged_obs, vision_obs, key, config
    )
    
    # Update Student
    updates, opt_state = tx.update(grads, opt_state, student_params)
    new_student_params = optax.apply_updates(student_params, updates)
    
    return new_student_params, opt_state, metrics
