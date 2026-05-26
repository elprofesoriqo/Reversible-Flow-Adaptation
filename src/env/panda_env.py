import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import mjx

@struct.dataclass
class PandaEnvState:
    mjx_data: mjx.Data
    target_object_pos: jax.Array
    target_object_mass: jax.Array
    
    # PD Reflex and Handoff State
    reflex_active: bool
    reflex_step_count: int
    rolling_action_buffer: jax.Array
    
class PandaEnv:
    def __init__(self, config):
        self.config = config
        
        # Load MJX model from menagerie asset
        self.mj_model = mujoco.MjModel.from_xml_path(config.asset_path)
        self.mj_model.opt.timestep = 0.02 # 50Hz control
        
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # PD Reflex specific gains (high-dampening, low-gain)
        self.kp = 50.0
        self.kd = 5.0
        self.tripwire_threshold = 15.0 # Nm
        
    def reset(self, key: jax.Array) -> PandaEnvState:
        mjx_data = mjx.make_data(self.mjx_model)
        
        # Initialize in home posture
        qpos_home = jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # Safe assignment of qpos (pad with zeros if model has more joints like gripper)
        full_qpos = jnp.zeros(self.mjx_model.nq)
        # Handle cases where nq might be smaller (e.g., rigid hand)
        n_joints = min(7, self.mjx_model.nq)
        full_qpos = full_qpos.at[:n_joints].set(qpos_home[:n_joints])
        
        mjx_data = mjx_data.replace(qpos=full_qpos, qvel=jnp.zeros(self.mjx_model.nv))
        
        # We mock the target object for the Pick task
        target_pos = jax.random.uniform(key, (3,), minval=-0.2, maxval=0.2) + jnp.array([0.5, 0.0, 0.2])
        target_mass = jnp.array([1.0]) # Nominal mass 1kg
        
        buffer = jnp.zeros((25, self.config.action_dim))
        
        return PandaEnvState(
            mjx_data=mjx_data,
            target_object_pos=target_pos,
            target_object_mass=target_mass,
            reflex_active=jnp.array(False),
            reflex_step_count=jnp.array(0),
            rolling_action_buffer=buffer
        )
        
    def get_proprioceptive_obs(self, state: PandaEnvState) -> jax.Array:
        # qpos (7) + qvel (7)
        qpos = state.mjx_data.qpos[:7]
        qvel = state.mjx_data.qvel[:7]
        return jnp.concatenate([qpos, qvel])
        
    def get_privileged_obs(self, state: PandaEnvState) -> jax.Array:
        # Mock friction=0.5
        return jnp.concatenate([state.target_object_mass, jnp.array([0.5])])
        
    def get_vision_obs(self, state: PandaEnvState) -> jax.Array:
        batch_dims = state.mjx_data.qpos.shape[:-1]
        return jnp.zeros(batch_dims + self.config.vision_resolution)
        
    def _pd_reflex_policy(self, state: PandaEnvState) -> jax.Array:
        # High-dampening PD stabilizer toward current safe posture
        current_qpos = state.mjx_data.qpos[:7]
        current_qvel = state.mjx_data.qvel[:7]
        target_qpos = current_qpos # Freeze in place
        
        tau = self.kp * (target_qpos - current_qpos) - self.kd * current_qvel
        return tau
        
    def step(self, state: PandaEnvState, action: jax.Array) -> tuple[PandaEnvState, jax.Array, jnp.bool_, dict]:
        # action is assumed to be raw joint torques for the 7 DoF arm
        measured_torques = state.mjx_data.qfrc_actuator[:7]
        torque_error = jnp.linalg.norm(action - measured_torques)
        
        is_ood = torque_error > self.tripwire_threshold
        
        # Reflex Handoff Logic
        should_trigger_reflex = jnp.logical_or(state.reflex_active, is_ood)
        
        def true_fn():
            # Apply PD Reflex
            safe_action = self._pd_reflex_policy(state)
            new_count = state.reflex_step_count + 1
            # Reflex expires after 10 steps (200ms)
            keep_active = new_count < 10
            return safe_action, keep_active, new_count
            
        def false_fn():
            # Nominal generative action
            return action, jnp.array(False), jnp.array(0)
            
        applied_action, reflex_active, reflex_count = jax.lax.cond(
            should_trigger_reflex, true_fn, false_fn
        )
        
        # Buffer update
        new_buffer = jnp.roll(state.rolling_action_buffer, shift=-1, axis=0)
        new_buffer = new_buffer.at[-1].set(applied_action)
        
        # Apply to MJX
        full_ctrl = jnp.zeros(self.mjx_model.nu)
        n_ctrl = min(7, self.mjx_model.nu)
        full_ctrl = full_ctrl.at[:n_ctrl].set(applied_action[:n_ctrl])
        mjx_data = state.mjx_data.replace(ctrl=full_ctrl)
        
        mjx_data = mjx.step(self.mjx_model, mjx_data)
        
        next_state = state.replace(
            mjx_data=mjx_data,
            reflex_active=reflex_active,
            reflex_step_count=reflex_count,
            rolling_action_buffer=new_buffer
        )
        
        # Simple task reward: distance to target (using end effector heuristic xpos[-1])
        ee_pos = mjx_data.xpos[-1] 
        dist = jnp.linalg.norm(ee_pos - state.target_object_pos)
        reward = -dist
        
        # Done condition
        done = dist > 2.0 # Out of bounds
        
        info = {
            'is_ood': is_ood,
            'torque_error': torque_error,
            'a_corr': applied_action
        }
        return next_state, reward, done, info
