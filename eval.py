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
import ray
from cluster2 import TrafficLightEnv, TrafficLightModel
def evaluate():
    # Initialize Ray
    ray.init(local_mode=True, num_cpus=2)

    # Register the custom model
    ModelCatalog.register_custom_model("my_traffic_model", TrafficLightModel)

    # Load the trained model
    latest_checkpoint = os.path.abspath(r"C:\Users\pc\Documents\Trafic\newRes\PPO_2025-01-24_03-51-14\PPO_TrafficLightEnv_141fd_00000_0_2025-01-24_03-51-15\checkpoint_000108")
    trained_agent = PPO.from_checkpoint(latest_checkpoint)

    # Create the SUMO environment
    env_config = {
        "num_lights": 5,
        "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\Sim03\osm.sumocfg",
        "max_steps": 20000,
    }
    env = TrafficLightEnv(env_config)

    for episode in range(10000):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            try:
                action = trained_agent.compute_single_action(obs, explore=False)
                print(f"Episode {episode+1}, Step {step}, Action: {action}, Observation: {obs}")
            except Exception as e:
                print(f"Error during action computation: {e}")
                break

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                done = True
                print(f"Episode {episode+1} finished with total reward: {total_reward} in {step} steps.")

    traci.close()
    ray.shutdown()

if __name__ == "__main__":
    evaluate()