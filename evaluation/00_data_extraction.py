import wandb
import pandas as pd
import os

project_name = "agile-locomotion-flow"
entity = None

print(f"Connecting to WandB Project: {project_name}...")
api = wandb.Api()

path = project_name if entity is None else f"{entity}/{project_name}"
runs = api.runs(path)

print(f"Found {len(runs)} runs. Extracting histories...")

os.makedirs('data', exist_ok=True)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # Extract metadata
    summary_list.append(run.summary._json_dict)
    config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})
    name_list.append(run.name)
    
    # Extract full history for learning curves
    print(f"  Downloading history for run: {run.name}")
    # Pulling 1000 samples to ensure high-fidelity learning curves
    history = run.history(samples=1000) 
    history['run_name'] = run.name
    
    history.to_csv(f"data/{run.name}_history.csv", index=False)
    
# Create a summary dataframe
summary_df = pd.DataFrame(summary_list)
summary_df['run_name'] = name_list

# Extract useful config params into summary
for i, config in enumerate(config_list):
    for key in ['num_envs', 'action_dim', 'asset_path', 'batch_size', 'chunk_size', 'num_epochs', 'learning_rate', 'vit_num_heads', 'wandb_project', 'vit_num_layers', 'vit_patch_size', 'aux_loss_weight', 'buffer_capacity', 'diffusion_steps', 'obs_dim_proprio', 'vision_feat_dim', 'vision_resolution', 'obs_dim_privileged']:
        if key in config:
            val = config[key]
            if isinstance(val, (list, tuple)):
                val = str(val)
            summary_df.loc[i, key] = val

summary_df.to_csv('data/wandb_runs_summary.csv', index=False)
print("Data extraction complete. Saved to data/")
