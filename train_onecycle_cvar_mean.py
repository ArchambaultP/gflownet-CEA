"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import open_dict
import wandb
from gflownet.utils.common import gflownet_from_config


@hydra.main(config_path="./config", config_name="train_onecycle_cvar_mean", version_base="1.1")
def main(config):

    # Set and print working and logging directory
    with open_dict(config):
        config.logger.logdir.path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)



    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config)
    # _original_end = gflownet.logger.end

    # def _end_with_timeout():
    #     import threading
    #     # Force exit if wandb.finish() takes more than 60s
    #     timer = threading.Timer(60.0, lambda: os._exit(0))
    #     timer.daemon = True
    #     timer.start()
    #     _original_end()
    #     timer.cancel()

    # gflownet.logger.end = _end_with_timeout

    import wandb
    if wandb.run is not None:
        wandb.run.config.update({
            "step_fraction": os.environ.get("STEP_FRACTION", "unknown"),
            "reward_cache": os.environ.get("REWARD_CACHE_PATH", "none"),
        })
    # Train GFlowNet
    gflownet.train()

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)
    
    gflownet.proxy.save_final_cache()
    print("Shutting Down")
    try:
        gflownet.proxy.pool.shutdown()
        print("Shutdown Pool")
    except Exception as e:
        print(f"Exception: {e}")
        pass


    # Close logger
    # TODO: make it gflownet.end() - perhaps there are other things to end
    gflownet.logger.end()

def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
