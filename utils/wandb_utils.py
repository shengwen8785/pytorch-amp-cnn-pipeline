import wandb

def initialize_wandb(project: str, name: str, config: dict):
    wandb.login()
    wandb.init(project=project, name=name, config=config)