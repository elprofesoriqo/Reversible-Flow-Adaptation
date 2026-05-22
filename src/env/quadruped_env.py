# pyrefly: ignore [missing-import]
import jax
# pyrefly: ignore [missing-import]
import jax.numpy as jnp
# pyrefly: ignore [missing-import]
import mujoco
# pyrefly: ignore [missing-import]
from mujoco import mjx
import os
import subprocess

class QuadrupedEnv:
    """
    MuJoCo MJX environment wrapper for the Unitree Go2.
    """
    
    def __init__(self, config):
        self.config = config
        self._ensure_assets_exist()
        
        self.mj_model = mujoco.MjModel.from_xml_path(self.config.asset_path)
        self.mj_model.opt.timestep = 0.002
        
        for i in range(self.mj_model.ngeom):
            if self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
        
        self.mjx_model = mjx.put_model(self.mj_model)
        
    def _ensure_assets_exist(self):
        """Fetches the mujoco_menagerie if the Go2 model is missing."""
        asset_dir = os.path.dirname(self.config.asset_path)
        if not os.path.exists(self.config.asset_path):
            print("Downloading Unitree Go2 assets from mujoco_menagerie...")
            os.makedirs("src/assets", exist_ok=True)
            subprocess.run([
                "git", "clone", "--depth", "1", 
                "https://github.com/google-deepmind/mujoco_menagerie.git", 
                "src/assets/mujoco_menagerie"
            ], check=True)
            print("Assets downloaded successfully.")

    def reset(self, key: jax.Array) -> mjx.Data:
        """Resets the environment and returns the initial MJX data state."""
        mjx_data = mjx.put_data(self.mj_model, mujoco.MjData(self.mj_model))
        
        # Add some random noise to initial joint positions
        qpos_noise = jax.random.uniform(key, shape=(self.mjx_model.nq,), minval=-0.1, maxval=0.1)
        mjx_data = mjx_data.replace(qpos=self.mjx_model.qpos0 + qpos_noise)
        
        # Forward kinematics to update body positions
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        return mjx_data
        
    def step(self, data: mjx.Data, action: jax.Array) -> tuple[mjx.Data, jax.Array, jax.Array]:
        """Steps the MJX environment forward on the GPU and returns (data, reward, done)."""
        # Apply action to control signals
        data = data.replace(ctrl=action)
        # physics simulation
        data = mjx.step(self.mjx_model, data)
        
        # Compute Reward and Done
        reward = self._compute_reward(data, action)
        done = self._compute_done(data)
        
        return data, reward, done
        
    def _compute_reward(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        """Standard locomotion reward (Forward Velocity + Energy Penalty)."""
        forward_vel = data.qvel[0]
        # Target velocity = 1.0 m/s
        vel_reward = jnp.exp(-((forward_vel - 1.0) ** 2))
        
        # Energy penalty (minimize control effort)
        energy_penalty = 0.01 * jnp.sum(jnp.square(action))
        
        return vel_reward - energy_penalty
        
    def _compute_done(self, data: mjx.Data) -> jax.Array:
        """Termination condition (crashes)."""
        z_height = data.qpos[2]
        crashed = z_height < 0.2
        return crashed
        
    def get_proprioceptive_obs(self, data: mjx.Data) -> jax.Array:
        """Extracts available sensor data (joint pos/vel, IMU)."""
        # Joint positions (12) and velocities (12)
        qpos_joints = data.qpos[7:19]
        qvel_joints = data.qvel[6:18]
        
        # IMU Simulation: Projected Gravity
        q = data.qpos[3:7] # Root quaternion [w, x, y, z]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Rotation matrix from quaternion
        rot_mat = jnp.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
        ])
        # Project global gravity vector [0, 0, -1] into local base frame
        projected_gravity = rot_mat.T @ jnp.array([0.0, 0.0, -1.0])
        
        return jnp.concatenate([qpos_joints, qvel_joints, projected_gravity])
        
    def get_privileged_obs(self, data: mjx.Data) -> jax.Array:
        """Extracts oracle data (root linear and angular velocity)."""
        # Exact root linear velocity (3) and angular velocity (3)
        return data.qvel[0:6]
        
    def get_vision_obs(self, data: mjx.Data) -> jax.Array:
        """Computes a spatial depth map of the terrain."""
        # Grid limits (e.g. 1 meter around the robot)
        grid_size = self.config.vision_resolution[0]
        xs = jnp.linspace(-1.0, 1.0, grid_size)
        ys = jnp.linspace(-1.0, 1.0, grid_size)
        X, Y = jnp.meshgrid(xs, ys)
        
        # Robot Height
        robot_z = data.qpos[2]
        
        # Apply orientation tilt (pitch/roll) to the depth calculation
        q = data.qpos[3:7]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Normal vector of the robot base in world frame
        normal_z = 1 - 2*x*x - 2*y*y
        normal_x = 2*x*z + 2*w*y
        normal_y = 2*y*z - 2*w*x
        
        # depth calculation for the flat plane
        # Depth = (robot_z + X * normal_x + Y * normal_y) / normal_z
        # We clip it to avoid singularities or negative depths if pointing up
        depth_map = (robot_z + X * normal_x + Y * normal_y) / (jnp.abs(normal_z) + 1e-6)
        depth_map = jnp.clip(depth_map, 0.0, 5.0)
        
        # Expand dims to match (64, 64, 1)
        return jnp.expand_dims(depth_map, axis=-1)
