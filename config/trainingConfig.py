
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

class MetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            episode_metrics = {
                "queue_lengths": [],
                "waiting_times": [],
                "throughputs": []
            }
            
            # Debug print
            #print("\nProcessing episode end metrics:")
            #print(f"Available agents: {episode.get_agents()}")
            
            for agent_id in episode.get_agents():
                info = episode.last_info_for(agent_id)
                #print(f"Agent {agent_id} info: {info}")
                
                if isinstance(info, dict):
                    # Safely extract metrics with explicit type conversion
                    queue_length = float(info.get("queue_length", 0.0))
                    waiting_time = float(info.get("waiting_time", 0.0))
                    throughput = float(info.get("throughput", 0.0))
                    
                    episode_metrics["queue_lengths"].append(queue_length)
                    episode_metrics["waiting_times"].append(waiting_time)
                    episode_metrics["throughputs"].append(throughput)
            
            # Log metrics only if we have data
            if episode_metrics["queue_lengths"]:
                mean_queue = np.mean(episode_metrics["queue_lengths"])
                episode.custom_metrics["mean_queue_length"] = mean_queue
                #print(f"Logged mean queue length: {mean_queue}")
                
            if episode_metrics["waiting_times"]:
                mean_wait = np.mean(episode_metrics["waiting_times"])
                episode.custom_metrics["mean_waiting_time"] = mean_wait
                #print(f"Logged mean waiting time: {mean_wait}")
                
            if episode_metrics["throughputs"]:
                mean_throughput = np.mean(episode_metrics["throughputs"])
                episode.custom_metrics["mean_throughput"] = mean_throughput
                #print(f"Logged mean throughput: {mean_throughput}")
                
        except Exception as e:
            print(f"Error in metrics callback: {str(e)}")
            import traceback
            traceback.print_exc()

def get_ma_training_config():
    """
    Returns an RLlib PPOConfig set up for multi-agent training with a single shared policy,
    using the old ModelV2 approach for a custom model.
    """
    # First, register our custom model
    ModelCatalog.register_custom_model("my_traffic_model", MyTrafficModel)

    # We'll create a temporary env instance to get the single-agent obs/action spaces
    # for the multi-agent policies dict.
    temp_env = TrafficLightMAEnv({"num_lights": 5})

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            env=TrafficLightMAEnv,
            env_config={
                "num_lights": 5,
                "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\simulation.sumocfg",
                "max_steps": 20000,
            }
        )
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2,
            rollout_fragment_length=200, 
            sample_timeout_s=180,
        )
        .training(
            train_batch_size=6000,
            num_epochs=10,
            gamma=0.99,
            lr=3e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        .callbacks(MetricsCallback)
        .reporting(
            metrics_num_episodes_for_smoothing=100,
            keep_per_episode_custom_metrics=True,
            min_sample_timesteps_per_iteration=1000,
            min_time_s_per_iteration=1,
            metrics_episode_collection_timeout_s=60,
        )
        .debugging(
            logger_config={
                "type": "ray.tune.logger.TBXLogger",
                "logdir": "./logs",
                "log_metrics_tables": True,
            }
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  
                    temp_env.single_light_obs_space,
                    temp_env.single_light_act_space,
                    {
                        "model": {
                            "custom_model": "my_traffic_model",
                            
                           
                        },
                    },
                )
            },
            policy_mapping_fn=my_policy_mapping,
        )

    )
    return config

