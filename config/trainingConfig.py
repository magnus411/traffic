
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air import RunConfig
import os
from environments.trafficEnv import TrafficLightMAEnv
from models.networks import CentralizedCriticRLModule
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian

from models.networks import MyTrafficModel
from environments.trafficEnv import TrafficLightMAEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import torch
from typing import Dict
import numpy as np

def my_policy_mapping(agent_id, episode=None, worker=None, **kwargs):
    return "shared_policy"
import torch



from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
import os

def get_single_agent_training_config():
  
    use_custom_model = True
    if use_custom_model:
        ModelCatalog.register_custom_model("my_traffic_model", TrafficLightModel)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=TrafficLightEnv,
            env_config={
                "num_lights": 5,
                "sumo_cfg": r"\\WIN-Q5NH2TGKKGM\RLTraffic\data\sumo\Sim03\osm.sumocfg",
                "max_steps": 18000,
            },
        )
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=20,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            sample_timeout_s=320,
        )
        .learners(
            num_learners=2,
            num_cpus_per_learner=1,
            num_gpus_per_learner=1,

        )

        .training(
            train_batch_size=16000,
            num_epochs=16,
            gamma=0.99,
            lr=1e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=20.0,
            entropy_coeff=0.01,
            kl_coeff=0.1,
            grad_clip=1.0,
        )
        .reporting(
            metrics_num_episodes_for_smoothing=100,
            keep_per_episode_custom_metrics=True,
            min_sample_timesteps_per_iteration=1000,
            min_time_s_per_iteration=1,
            metrics_episode_collection_timeout_s=120,
        )
        .debugging(
            logger_config={
                "type": "ray.tune.logger.TBXLogger",
                "logdir": "./logs",
                "log_metrics_tables": True,
            },
        )
    )

    if use_custom_model:
        config.training(
            model={
                "custom_model": "my_traffic_model",
            }
        )

    return config
