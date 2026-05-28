import jax
import jax.numpy as jnp
from typing import Tuple, Dict
import mujoco
from mujoco import mjx
import flax.struct

@flax.struct.dataclass
class EnvState:
    mjx_data: mjx.Data
    ood_flag: jax.Array
    reflex_count: jax.Array
    rolling_buffer: jax.Array
    
class QuadrupedEnv:
    def __init__(self, config):
        self.config = config
        self.action_dim = config.action_dim
        
        # Load real MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(config.asset_path)
        self.mj_model.opt.timestep = 0.01 
        
        # JAX compatible model
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # OOD Tripwire & Reflex params
        self.tau_trip_threshold = 1.5 # Torque L2 norm threshold
        self.pd_k_steps = 10 # 200ms handoff
        self.kp_reflex = 5.0 # Low gain
        self.kd_reflex = 0.5 # High dampening
        
        self.rolling_buffer_size = 25 # 500ms
        
    def reset(self, key: jax.Array) -> EnvState:
        """Returns initial state."""
        # Initialize default mjx.Data
        mj_data = mujoco.MjData(self.mj_model)
        mjx_data = mjx.put_data(self.mj_model, mj_data)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        
        ood_flag = jnp.array(False)
        reflex_count = jnp.array(0, dtype=jnp.int32)
        buffer = jnp.zeros((self.rolling_buffer_size, self.action_dim))
        
        return EnvState(
            mjx_data=mjx_data,
            ood_flag=ood_flag,
            reflex_count=reflex_count,
            rolling_buffer=buffer
        )
        
    def get_proprioceptive_obs(self, state: EnvState) -> jax.Array:
        # Assuming 12 joints, skip 7 free pos and 6 free vel
        qpos = state.mjx_data.qpos[..., 7:19]
        qvel = state.mjx_data.qvel[..., 6:18]
        proj_gravity = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), qpos[..., :3].shape)
        return jnp.concatenate([qpos, qvel, proj_gravity], axis=-1)
        
    def get_privileged_obs(self, state: EnvState) -> jax.Array:
        return state.mjx_data.qvel[..., 0:6]
        
    def get_vision_obs(self, state: EnvState) -> jax.Array:
        batch_dims = state.mjx_data.qpos.shape[:-1]
        return jnp.zeros(batch_dims + self.config.vision_resolution)
        
    def step(self, state: EnvState, action: jax.Array) -> Tuple[EnvState, jax.Array, jax.Array, Dict]:
        """
        Advances the environment by one step using mjx.
        """
        tau_cmd = action
        
        # Real measured torque from previous step's physics engine evaluation
        tau_measured = state.mjx_data.qfrc_actuator[..., 6:18]
        
        torque_error = jnp.linalg.norm(tau_cmd - tau_measured, axis=-1)
        tripwire_triggered = torque_error > self.tau_trip_threshold
        
        # Update OOD flag
        is_ood = jnp.logical_or(tripwire_triggered, state.reflex_count > 0)
        
        # PD Reflex Controller (High-dampening, low-gain)
        qpos_error = -state.mjx_data.qpos[..., 7:19]
        qvel_error = -state.mjx_data.qvel[..., 6:18]
        a_corr = self.kp_reflex * qpos_error + self.kd_reflex * qvel_error
        
        # Decide which action to apply
        applied_action = jnp.where(is_ood[..., None], a_corr, action)
        
        # Pad action back to full ctrl size if there are more than 12 actuators
        ctrl = jnp.pad(applied_action, ((0,),) * (applied_action.ndim - 1) + ((0, state.mjx_data.ctrl.shape[-1] - self.action_dim),))
        mjx_data = state.mjx_data.replace(ctrl=ctrl)
        
        # Step physics
        mjx_data = mjx.step(self.mjx_model, mjx_data)
        
        # Rolling Action Buffer
        new_buffer = jnp.roll(state.rolling_buffer, shift=-1, axis=-2)
        new_buffer = new_buffer.at[..., -1, :].set(applied_action)
        
        # Update reflex count
        new_reflex_count = jnp.where(
            tripwire_triggered & (state.reflex_count == 0), 
            self.pd_k_steps, 
            jnp.maximum(0, state.reflex_count - 1)
        )
        new_ood_flag = new_reflex_count > 0
        
        next_state = EnvState(
            mjx_data=mjx_data,
            ood_flag=new_ood_flag,
            reflex_count=new_reflex_count,
            rolling_buffer=new_buffer
        )
        
        reward = jnp.sum(applied_action, axis=-1)
        done = jnp.zeros_like(reward, dtype=bool)
        
        info = {
            'is_ood': new_ood_flag,
            'torque_error': torque_error,
            'a_corr': a_corr
        }
        
        return next_state, reward, done, info
