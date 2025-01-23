import os
import numpy as np
import gymnasium as gym
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
import traci
import time
from ray.tune import Tuner
from ray.rllib.algorithms.ppo import PPO

from single import TrafficLightEnv, TrafficLightModel
def evaluate():
    # Load the trained model
    config = get_single_agent_training_config()
    ModelCatalog.register_custom_model("my_traffic_model", TrafficLightModel)

    # Restore the best checkpoint
    latest_checkpoint = os.path.abspath(r"C:\Users\pc\Documents\Trafic\results\PPO_2025-01-22_18-57-13\PPO_TrafficLightEnv_befd4_00000_0_2025-01-22_18-57-13\checkpoint_000053")
    trained_agent = PPO.from_checkpoint(latest_checkpoint)

    # Create the SUMO environment
    env_config = {
        "num_lights": 5,
        "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\Sim02\osm.sumocfg",
        "max_steps": 20000,
    }
    env = TrafficLightEnv(env_config)

    # Run the evaluation
    obs, info = env.reset()
    done = False
    while not done:
        print("Observation:", obs)

        try:
            # Use the observation directly without flattening
            action = trained_agent.compute_single_action(obs)
            print("Action:", action)

        except Exception as e:
            print(f"Error during action computation: {e}")
            break

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    traci.close()

def get_single_agent_training_config():
    """
    Returns an RLlib PPOConfig set up for single-agent training.
    """
    # If a custom model is still needed, register it
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
                "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\Sim02\osm.sumocfg",
                "max_steps": 18000,
            },
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
            train_batch_size=12000,
            num_epochs=13,
            gamma=0.99,
            lr=1e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=30.0,
            entropy_coeff=0.01,
            kl_coeff=0.0,
            grad_clip=1.0,
        )
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
            },
        )
    )

    # Add a custom model if required
    if use_custom_model:
        config.training(
            model={
                "custom_model": "my_traffic_model",
            }
        )

    return config


if __name__ == "__main__":
    evaluate()