import wandb

# Initialize a new run
wandb.init(project="your_project_name")

# Use wandb.log to log metrics
wandb.log({"accuracy": 0.9})
