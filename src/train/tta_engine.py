import threading
import time
import jax
import jax.numpy as jnp
from src.train.tta_loss import tta_update_step

class TTAEngine:
    """
    Asynchronous Test-Time Adaptation Engine.
    Runs a 2Hz background thread that computes Hutchinson trace gradients
    and updates a shadow copy of the LoRA weights, which is safely swapped
    into the 50Hz main control thread.
    """
    def __init__(self, student_params, opt_state, projection_params, tx, student_apply_fn, projection_apply_fn):
        self.active_params = student_params
        self.shadow_params = student_params
        
        self.projection_params = projection_params
        
        self.opt_state = opt_state
        self.tx = tx
        
        self.student_apply = student_apply_fn
        self.projection_apply = projection_apply_fn
        
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # Buffer for latest OOD data
        self.latest_ood_data = None
        self.data_lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._async_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def trigger_ood_adaptation(self, proprio, vision, a_corr):
        """
        Called by the environment step function when OOD is detected.
        Passes the latest 500ms rolling buffer data to the adaptation thread.
        """
        with self.data_lock:
            self.latest_ood_data = (proprio, vision, a_corr)
            
    def swap_buffers(self):
        """
        Called by the 50Hz control thread to atomically swap to the latest adapted weights.
        """
        with self.lock:
            self.active_params = self.shadow_params
            
    def get_active_params(self):
        return self.active_params
        
    def _async_loop(self):
        """
        Runs continuously at 2Hz in the background.
        If there's new OOD data, it runs the Hutchinson backward pass and updates shadow weights.
        """
        target_dt = 0.5 # 2Hz
        
        while self.running:
            start_time = time.time()
            
            data_to_process = None
            with self.data_lock:
                if self.latest_ood_data is not None:
                    data_to_process = self.latest_ood_data
                    self.latest_ood_data = None
            
            if data_to_process is not None:
                proprio, vision, a_corr = data_to_process

                new_params, new_opt_state, metrics = tta_update_step(
                    self.shadow_params, 
                    self.opt_state, 
                    self.projection_params,
                    proprio, 
                    vision, 
                    a_corr, 
                    self.tx, 
                    self.student_apply, 
                    self.projection_apply
                )
                
                self.opt_state = new_opt_state
                
                # Write to shadow buffer
                with self.lock:
                    self.shadow_params = new_params
                    
                print(f"[TTA Engine] Adaptation Step Complete. NLL: {metrics['nll']:.4f} | LoRA L2: {metrics['lora_l2']:.4f}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0.0, target_dt - elapsed)
            time.sleep(sleep_time)
