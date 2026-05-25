import jax
import jax.numpy as jnp
import optax
from typing import Callable

def hutchinson_trace_estimator(v_fn, x, M=5, key=None):
    """
    Estimates Tr(dv/dx) using Hutchinson's trace estimator with M Rademacher samples.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
        
    keys = jax.random.split(key, M)
    
    def single_sample_trace(k):
        # Rademacher noise: uniform from {-1, 1}
        eps = jax.random.rademacher(k, shape=x.shape, dtype=x.dtype)
        # Compute jvp: eps^T * (dv/dx) = JVP(v, x, eps)
        _, v_jvp = jax.jvp(v_fn, (x,), (eps,))
        # Trace estimate for this sample is sum(eps * v_jvp) over feature dims
        return jnp.sum(eps * v_jvp, axis=tuple(range(1, eps.ndim)))
        
    traces = jax.vmap(single_sample_trace)(keys)
    return jnp.mean(traces, axis=0) # Average over M samples

def rk4_backward_step_with_trace(v_fn, x_t, t, dt, key):
    """
    Takes a single RK4 step backward from t to t - dt, 
    and integrates the trace of the Jacobian.
    """
    def v_and_trace(x_curr, t_curr, subkey):
        v = v_fn(x_curr, t_curr)
        tr = hutchinson_trace_estimator(lambda x: v_fn(x, t_curr), x_curr, M=5, key=subkey)
        return v, tr
        
    k1_key, k2_key, k3_key, k4_key = jax.random.split(key, 4)
    
    k1_v, k1_tr = v_and_trace(x_t, t, k1_key)
    k2_v, k2_tr = v_and_trace(x_t - 0.5 * dt * k1_v, t - 0.5 * dt, k2_key)
    k3_v, k3_tr = v_and_trace(x_t - 0.5 * dt * k2_v, t - 0.5 * dt, k3_key)
    k4_v, k4_tr = v_and_trace(x_t - dt * k3_v, t - dt, k4_key)
    
    x_prev = x_t - (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    delta_log_p = (dt / 6.0) * (k1_tr + 2*k2_tr + 2*k3_tr + k4_tr)
    
    return x_prev, delta_log_p

def integrate_flow_backward(v_fn_time, x_1, num_steps=10, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
        
    dt = 1.0 / num_steps
    
    def body_fn(val, step_key):
        x_t, log_p, t = val
        x_prev, delta_log_p = rk4_backward_step_with_trace(v_fn_time, x_t, t, dt, step_key)
        return (x_prev, log_p + delta_log_p, t - dt), None
        
    keys = jax.random.split(key, num_steps)
    (x_0, total_delta_log_p, _), _ = jax.lax.scan(body_fn, (x_1, jnp.zeros(x_1.shape[0]), 1.0), keys)
    
    return x_0, total_delta_log_p

def compute_tta_loss(student_params, projection_params, proprio, vision, a_corr, student_apply_fn, projection_apply_fn, weight_decay=1e-4):
    """
    Computes L_TTA: negative log-likelihood of a_corr + LoRA L2 penalty.
    """
    # Project a_corr (PD torque) to generative manifold
    # projection_apply_fn map (B, Chunk, Action) -> (B, Chunk, Action)
    a_proj, proj_log_det = projection_apply_fn(projection_params, a_corr, reverse=False, return_log_det=True)
    
    # ODE Backward Integration
    def v_fn_time(x, t):
        t_arr = jnp.full((x.shape[0], 1), t)
        # student_apply_fn returns vector_field, priv_pred, physics_pred
        v, _, _ = student_apply_fn(student_params, proprio, vision, x, t_arr)
        return v
        
    z_corr, delta_log_p = integrate_flow_backward(v_fn_time, a_proj, num_steps=10)
    
    log_p0 = -0.5 * jnp.sum(jnp.square(z_corr), axis=tuple(range(1, z_corr.ndim)))
    log_p_a = log_p0 + delta_log_p + proj_log_det
    nll = -jnp.mean(log_p_a)
    
    # L2 Weight Decay on LoRA parameters
    flat_params, _ = jax.tree_util.tree_flatten_with_path(student_params)
    lora_l2_norm = 0.0
    for path, value in flat_params:
        path_str = ''.join(str(p) for p in path)
        if 'lora' in path_str.lower():
            lora_l2_norm += jnp.sum(jnp.square(value))
            
    loss = nll + weight_decay * lora_l2_norm
    
    return loss, {'nll': nll, 'lora_l2': lora_l2_norm, 'loss': loss}

@jax.jit
def tta_update_step(student_params, opt_state, projection_params, proprio, vision, a_corr, tx, student_apply_fn, projection_apply_fn):
    grad_fn = jax.value_and_grad(compute_tta_loss, argnums=0, has_aux=True)
    (loss, metrics), grads = grad_fn(student_params, projection_params, proprio, vision, a_corr, student_apply_fn, projection_apply_fn)
    
    # Zero out gradients for non-LoRA parameters
    def mask_non_lora_grads(path, grad):
        path_str = ''.join(str(p) for p in path)
        if 'lora' in path_str.lower():
            return grad
        else:
            return jnp.zeros_like(grad)
            
    filtered_grads = jax.tree_util.tree_map_with_path(mask_non_lora_grads, grads)
    
    updates, opt_state = tx.update(filtered_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, opt_state, metrics
