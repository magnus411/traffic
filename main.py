import os


os.environ["FORCE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3"

import glob
import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig  # Added CheckpointConfig import
from ray.tune import Tuner

from config.trainingConfig import get_ma_training_config
from ray.rllib.models import ModelCatalog
from models.networks import MyTrafficModel

def train():
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=1)
    print(ray.get_gpu_ids())
    print(ray.available_resources())

    config = get_ma_training_config()

    # Define the storage path where checkpoints are saved
    storage_path = os.path.abspath("./results")
    os.makedirs(storage_path, exist_ok=True)

    # Find the latest checkpoint if resuming
    latest_checkpoint = os.path.abspath(
    "C:/Users/pc/Documents/Trafic/results/PPO_2025-01-21_09-10-43/"
    )

    
    if latest_checkpoint:
        # Restore a new Tuner instance from the checkpoint
        print("Restoring from checkpoint...")
        tuner = Tuner.restore(
            latest_checkpoint, 
            "PPO",
            param_space=config,
            
        )
    else:
        tuner = Tuner(
            "PPO",
            param_space=config,
            run_config=RunConfig(
                storage_path=storage_path,
                stop={"training_iteration": 100},  # Define your stopping criteria
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=3,  # Save every 3 iterations
                    checkpoint_at_end=True,
                ),
            ),
        )


    results = tuner.fit()

    # Start the training

    # Retrieve the best result
    best_result = results.get_best_result(metric="mean_reward", mode="max")
    best_checkpoint = best_result.checkpoint
    print("Best checkpoint:", best_checkpoint)

if __name__ == "__main__":
    train()