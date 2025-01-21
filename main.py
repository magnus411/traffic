import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig  # Added CheckpointConfig import
from ray.tune import Tuner
import os
import glob

from config.trainingConfig import get_ma_training_config
from ray.rllib.models import ModelCatalog
from models.networks import MyTrafficModel

def train():
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=1)

    config = get_ma_training_config()

    # Define the storage path where checkpoints are saved
    storage_path = os.path.abspath("./results")
    os.makedirs(storage_path, exist_ok=True)

    # Find the latest checkpoint if resuming
    latest_checkpoint = None
    checkpoint_files = glob.glob(os.path.join(storage_path, "*", "checkpoint_*"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    
    # Configure the Tuner
    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            storage_path=storage_path,
            stop={"training_iteration": 100},  # Define your stopping criteria
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=3,  # Yes, this will save every iteration
                checkpoint_at_end=True,
            ),
            # restore_from_path=latest_checkpoint if latest_checkpoint else None,
        ),
    )

    # Start the training
    results = tuner.fit()

    # Retrieve the best result
    best_result = results.get_best_result(metric="mean_reward", mode="max")
    best_checkpoint = best_result.checkpoint
    print("Best checkpoint:", best_checkpoint)

if __name__ == "__main__":
    train()