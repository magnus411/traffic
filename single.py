
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TrafficLightEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.num_lights = config.get("num_lights", 5)
        self.light_ids = config.get("light_ids", ["A0", "B0", "C0", "D0", "E0"][: self.num_lights])

        self.sumo_cfg = config.get("sumo_cfg", "data/sumo/simulation.sumocfg")
        self.max_steps = config.get("max_steps", 10000)

        self._is_connected = False
        self.current_step = 0

        # Combined observation space
        self.observation_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "local_state": gym.spaces.Dict({
                    "queue_length": gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
                    "current_phase": gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
                    "waiting_time": gym.spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32),
                    "lane_density": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                    "mean_speed": gym.spaces.Box(low=0, high=30, shape=(4,), dtype=np.float32),
                }),
            }) for light_id in self.light_ids
        })

        # Combined action space
        # Combined action space
        self.action_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "phase": gym.spaces.Discrete(3),
                "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
            }) for light_id in self.light_ids
        })

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        self._start_simulation()
        self.current_step = 0

        for _ in range(10):  # Warm-up steps
            traci.simulationStep()

        obs = {light_id: self._get_observation(light_id) for light_id in self.light_ids}
        # Create an empty info dict
        info = {}
        
        # Return just observation and info
        return obs, info
    def step(self, action):
        """Execute one step of the environment."""
        self._ensure_connection()
        self.current_step += 1

        for light_id, light_action in action.items():
            phase = int(light_action["phase"])
            # Fix the deprecation warning by extracting single value
            duration = float(light_action["duration"].item())  # Use .item() to convert to scalar
            duration = max(5.0, min(60.0, duration))

            traci.trafficlight.setPhase(light_id, phase)
            traci.trafficlight.setPhaseDuration(light_id, duration)

        traci.simulationStep()

        obs = {light_id: self._get_observation(light_id) for light_id in self.light_ids}
        rewards = sum(self._compute_reward(light_id) for light_id in self.light_ids)
        
        # Split 'done' into terminated and truncated
        terminated = traci.simulation.getMinExpectedNumber() <= 0  # Environment-determined termination
        truncated = self.current_step >= self.max_steps  # Truncation due to max steps
        
        info = {
            "episode_step": self.current_step,
            "total_vehicles": traci.vehicle.getIDCount(),
            "simulation_time": traci.simulation.getTime(),
        }

        return obs, rewards, terminated, truncated, info
    def _get_observation(self, light_id):
        """Get the observation for a specific traffic light."""
        lanes = traci.trafficlight.getControlledLanes(light_id)
        
        queue_length = self._get_queue_length(light_id)
        waiting_time = self._get_waiting_time(light_id)
        current_phase = np.array([traci.trafficlight.getPhase(light_id) % 3], dtype=np.float32)
        
        # Add these new observations
        lane_density = np.zeros(4, dtype=np.float32)
        mean_speed = np.zeros(4, dtype=np.float32)
        
        for i in range(min(4, len(lanes))):
            lane_id = lanes[i]
            try:
                # Get density (vehicles per meter)
                lane_length = traci.lane.getLength(lane_id)
                num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                lane_density[i] = num_vehicles / lane_length if lane_length > 0 else 0
                
                # Get mean speed
                mean_speed[i] = traci.lane.getLastStepMeanSpeed(lane_id)
            except traci.exceptions.TraCIException:
                pass
        
        return {
            "local_state": {
                "queue_length": queue_length,
                "current_phase": current_phase,
                "waiting_time": waiting_time,
                "lane_density": lane_density,
                "mean_speed": mean_speed
            }
        }
    def _compute_reward(self, light_id):
        """
        Compute reward for a specific traffic light.
        Rewards efficient traffic flow and penalizes congestion and waiting.
        """
        # Base metrics
        queue_length = np.sum(self._get_queue_length(light_id))
        waiting_time = np.sum(self._get_waiting_time(light_id)) / 100.0
        
        # Get controlled lanes
        lanes = traci.trafficlight.getControlledLanes(light_id)[:4]
        
        # Throughput: number of vehicles that passed through
        throughput = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
        
        # Speed efficiency: ratio of current speed to max speed
        speed_scores = []
        for lane in lanes:
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            if max_speed > 0:
                speed_scores.append(mean_speed / max_speed)
        speed_efficiency = np.mean(speed_scores) if speed_scores else 0
        
        # Density penalty: penalize high density which indicates congestion
        density_penalty = 0
        for lane in lanes:
            length = traci.lane.getLength(lane)
            num_vehicles = traci.lane.getLastStepVehicleNumber(lane)
            if length > 0:
                density = num_vehicles / length
                density_penalty += density
        
        # Emergency stop penalty: penalize emergency stops which indicate poor timing
        emergency_stops = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        
        # Combine all factors into final reward
        reward = (
            throughput * 2.0 +                    # Reward for throughput
            speed_efficiency * 5.0 -              # Reward for maintaining speed
            queue_length * 0.5 -                  # Penalty for queues
            waiting_time * 0.3 -                  # Penalty for waiting time
            density_penalty * 2.0 -               # Penalty for congestion
            emergency_stops * 1.0                 # Penalty for emergency stops
        )
        
        # Normalize reward to avoid extreme values
        reward = np.clip(reward, -10.0, 10.0)
        
        return reward
    # Helper methods (unchanged)
    def _start_simulation(self):
        """Start SUMO simulation with proper error handling"""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass  # Ignore errors when closing
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                sumo_cmd = [
                    "sumo",
                    "-c", self.sumo_cfg,
                    "--start",
                    "--quit-on-end", "true",
                    "--random",
                    "--time-to-teleport", "-1",
                    "--waiting-time-memory", "300",
                    "--no-warnings", "true"  # Reduce warning spam
                ]
                
                traci.start(sumo_cmd)
                self._is_connected = True
                break
                
            except Exception as e:
                print(f"SUMO connection attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                time.sleep(1)  # Wait before retry
                
        if not self._is_connected:
            raise RuntimeError("Failed to establish SUMO connection after multiple attempts")

    def _ensure_connection(self):
        """Ensure SUMO connection is active"""
        if not self._is_connected:
            self._start_simulation()


    def _get_queue_length(self, light_id):
        lanes = traci.trafficlight.getControlledLanes(light_id)
        result = np.zeros(4, dtype=np.float32)  # Explicitly use float32
        for i in range(min(4, len(lanes))):
            lane_id = lanes[i]
            try:
                halt_num = traci.lane.getLastStepHaltingNumber(lane_id)
                result[i] = np.float32(halt_num)  # Convert to float32
            except traci.exceptions.TraCIException:
                pass
        return result

    def _get_waiting_time(self, light_id):
        lanes = traci.trafficlight.getControlledLanes(light_id)
        result = np.zeros(4, dtype=np.float32)  # Explicitly use float32
        for i in range(min(4, len(lanes))):
            lane_id = lanes[i]
            try:
                wtime = traci.lane.getWaitingTime(lane_id)
                result[i] = np.float32(wtime)  # Convert to float32
            except traci.exceptions.TraCIException:
                pass
        return result


class TrafficLightModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Number of traffic lights
        self.num_lights = len(obs_space.original_space.spaces)
        
        # Define sub-model for processing each traffic light's local state
        self.light_state_encoder = nn.Sequential(
            nn.LayerNorm(20),  # 4 + 1 + 4 + 4 + 4 = 20 features
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Calculate combined feature size
        combined_size = 32 * self.num_lights
        
        # Global feature combiner
        self.global_combiner = nn.Sequential(
            nn.LayerNorm(combined_size),  # Add normalization
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            nn.Tanh(),  # Add output activation
        )
        
        # Separate value branch
        self.value_branch = nn.Sequential(
            nn.LayerNorm(combined_size),  # Add normalization
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        light_features = []
        for light_id in obs:
            local_state = obs[light_id]["local_state"]
            
            # Convert to tensors and normalize
            queue_length = torch.as_tensor(local_state["queue_length"], device=self.device).float()
            waiting_time = torch.as_tensor(local_state["waiting_time"], device=self.device).float() 
            current_phase = torch.as_tensor(local_state["current_phase"], device=self.device).float() 
            lane_density = torch.as_tensor(local_state["lane_density"], device=self.device).float()
            mean_speed = torch.as_tensor(local_state["mean_speed"], device=self.device).float()
            
            # Ensure current_phase is the right shape before expanding
            if current_phase.dim() == 1:
                current_phase = current_phase.unsqueeze(-1)
            
            # Expand current_phase from [batch_size, 1] to [batch_size, 4]
            current_phase = current_phase.expand(-1, 4)
            
            # Concatenate features
            light_obs = torch.cat([
                queue_length,      # [batch_size, 4]
                current_phase,     # [batch_size, 4]
                waiting_time,      # [batch_size, 4]
                lane_density,      # [batch_size, 4]
                mean_speed        # [batch_size, 4]
            ], dim=-1)  # Total shape: [batch_size, 20]
            light_features.append(self.light_state_encoder(light_obs))
        
        # Combine features from all lights
        combined_features = torch.cat(light_features, dim=-1)
        
        # Store features for value function
        self._features = combined_features
        
        # Compute action logits
        logits = self.global_combiner(combined_features)
        
        # Compute value function
        self._value_out = self.value_branch(combined_features).squeeze(1)
        
        return logits, state

    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out

    @property
    def device(self):
        return next(self.parameters()).device
    

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
                "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\simulation.sumocfg",
                "max_steps": 20000,
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
            train_batch_size=6000,
            num_epochs=10,
            gamma=0.99,
            lr=3e-4,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
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


from ray.air import RunConfig, CheckpointConfig  # Added CheckpointConfig import


def train():
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=1)
    print(ray.get_gpu_ids())
    print(ray.available_resources())

    config = get_single_agent_training_config()

        # Define the storage path where checkpoints are saved
    storage_path = os.path.abspath("./results")
    os.makedirs(storage_path, exist_ok=True)

        # Find the latest checkpoint if resuming
        #latest_checkpoint = os.path.abspath(
        #"C:/Users/pc/Documents/Trafic/results/PPO_2025-01-21_09-10-43/"
        #)

    latest_checkpoint = None
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