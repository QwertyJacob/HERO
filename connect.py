import wandb
import importlib


# Login to wandb
wandb.login()

def reload_modules(list_of_modules):
    for module in list_of_modules:
        print(f'reloading module :{module}')
        importlib.reload(module)