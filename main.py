
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.train.torch import TorchConfig, TorchTrainer
from ray.train import ScalingConfig

import ray
from ray import tune
from ray.tune import Tuner
from ray.air import RunConfig

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

import traci
import time
from collections import defaultdict
from typing import Dict, List, Tuple


from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from config.trainingConfig import get_single_agent_training_config



from ray.air import RunConfig, CheckpointConfig  



def train():
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True  
    ray.init(address="auto") 
    #ray.init(local_mode=True, num_cpus=2)


    config = get_single_agent_training_config()

    storage_path = r"\\WIN-Q5NH2TGKKGM\RLTraffic\ClusterResults"
    os.makedirs(storage_path, exist_ok=True)

    latest_checkpoint = None

    if latest_checkpoint:
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
                stop={"training_iteration": 800}, 
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=3,  
                    checkpoint_at_end=True,
                ),

                callbacks=[
                    WandbLoggerCallback(
                        project="Traffic", api_key="f16cce37f90a1ffe6dc3741f0f86df10cc04baed", log_config=True
                )
        ]

            ),

        )       

    results = tuner.fit()

    # Retrieve the best result
    best_result = results.get_best_result(metric="mean_reward", mode="max")
    best_checkpoint = best_result.checkpoint
    print("Best checkpoint:", best_checkpoint)

