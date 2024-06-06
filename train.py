import torch
import hydra
from omegaconf import DictConfig

from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="base")
def train(args: DictConfig):
    set_seed(args.seed)

    # ------------------
    #    Dataloader
    # ------------------

    # ------------------
    #       Model
    # ------------------

    # ------------------
    #  Loss & optimizer
    # ------------------

    # ------------------
    #   Start training
    # ------------------


if __name__ == "__main__":
    train()
