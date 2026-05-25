import os
import wandb
import numpy as np
import mujoco

class WandbLogger:
    def __init__(self, config):
        self.config = config
        
        local_dir = os.path.join(os.getcwd(), 'wandb_local')
        os.makedirs(local_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = local_dir
        
        wandb.init(
            project=config.wandb_project,
            config=config.__dict__,
            dir=local_dir
        )
        
        try:
            self.model = mujoco.MjModel.from_xml_path(config.asset_path)
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception as e:
            print(f"Warning: Renderer initialization failed: {e}")
            self.renderer = None

    def log_metrics(self, epoch, ppo_metrics, fm_metrics, sys_metrics):
        """Logs standard scalars and granular components."""
        log_dict = {"epoch": epoch}
        
        for k, v in ppo_metrics.items():
            log_dict[f"ppo/{k}"] = v
            
        granular_keys = ['forward_velocity', 'energy_penalty', 'posture_penalty', 'foot_clearance', 'mean_joint_torque', 'action_smoothness']
        for k in granular_keys:
            if k in ppo_metrics:
                log_dict[f"physics_and_rewards/{k}"] = ppo_metrics[k]
                
        for k, v in fm_metrics.items():
            log_dict[k] = v # fm_metrics already prefixed in distillation.py
            
        for k, v in sys_metrics.items():
            log_dict[f"system/{k}"] = v
            
        wandb.log(log_dict)
        
    def render_and_log_video(self, rollout_mjx_data_list, fps=50):
        """
        Takes a list of mjx.Data states from a rollout, renders them to video, 
        and logs to WandB as wandb.Video.
        """
        if self.renderer is None or not rollout_mjx_data_list:
            return
            
        frames = []
        mj_data = mujoco.MjData(self.model)
        
        import mujoco.mjx
        for mjx_data in rollout_mjx_data_list:
            mujoco.mjx.get_data(self.model, mj_data, mjx_data)
            self.renderer.update_scene(mj_data)
            pixels = self.renderer.render()
            frames.append(pixels)
            
        frames = np.stack(frames) # (T, H, W, C)
        
        # W&B expects (T, C, H, W)
        frames_wandb = np.transpose(frames, (0, 3, 1, 2))
        
        wandb.log({"media/rollout_video": wandb.Video(frames_wandb, fps=fps, format="mp4")})
        
    def log_vision_heatmap(self, vision_obs):
        """Logs a single 2D depth map as an image to verify vision inputs."""
        if vision_obs.ndim == 3 and vision_obs.shape[-1] == 1:
            img = vision_obs[..., 0]
        else:
            img = vision_obs
            
        wandb.log({"media/vision_depth_heatmap": wandb.Image(np.array(img), caption="Vision Depth Map")})
        
    def finish(self):
        wandb.finish()
