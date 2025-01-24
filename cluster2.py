
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

class TrafficLightEnv(gym.Env):
    def _create_observation_space(self):
        max_default_phases = 10
        return gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "local_state": gym.spaces.Dict({
                    "queue_length": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(120), 
                        shape=(4,), 
                        dtype=np.float32
                    ),
                    "current_phase": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(max_default_phases-1), 
                        shape=(1,), 
                        dtype=np.float32
                    ),
                    "phase_state": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(1), 
                        shape=(4,), 
                        dtype=np.float32
                    ),
                    "waiting_time": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(1e3), 
                        shape=(4,), 
                        dtype=np.float32
                    ),
                    "mean_speed": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(30), 
                        shape=(4,), 
                        dtype=np.float32
                    ),
                    "trip_penalty": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(60), 
                        shape=(1,), 
                        dtype=np.float32
                    ),
                    "idle_time": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(1e3), 
                        shape=(1,), 
                        dtype=np.float32
                    ),
                }),
                "downstream_context": gym.spaces.Dict({
                    "total_downstream_queue": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(400), 
                        shape=(1,), 
                        dtype=np.float32
                    ),
                    "avg_time_to_switch": gym.spaces.Box(
                        low=np.float32(0), 
                        high=np.float32(120), 
                        shape=(1,), 
                        dtype=np.float32
                    ),
                }),
            }) for light_id in self.light_ids
        })

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.num_lights = self.config.get("num_lights", 5)
        self.light_ids = self.config.get("light_ids", [
            "cluster_155251477_4254141700_6744555967_78275",
            "cluster_1035391877_6451996796",
            "cluster_163115359_2333391359_2333391361_78263_#1more",
            "cluster_105371_4425738061_4425738065_4425738069",
            "cluster_105372_4916088731_4916088738_4916088744"][:self.num_lights])
        self.sumo_cfg = self.config.get("sumo_cfg", "C:/Users/support/Desktop/Sim03/osm.sumocfg")
        self.max_steps = self.config.get("max_steps", 10000)
        
        # Cache frequently accessed data
        self._controlled_lanes_cache = {}
        self._downstream_lights_cache = {}
        self._lane_length_cache = {}
        self._max_speed_cache = {}
        
        # Initialize spaces (same as before)
        max_default_phases = 10
        self.observation_space = self._create_observation_space()
        self.action_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "phase": gym.spaces.Discrete(max_default_phases),
                "duration": gym.spaces.Box(
                    low=np.float32(5.0), 
                    high=np.float32(60.0), 
                    shape=(1,), 
                    dtype=np.float32
                ),
            }) for light_id in self.light_ids
        })

        
        self.action_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "phase": gym.spaces.Discrete(max_default_phases),
                "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
            }) for light_id in self.light_ids
        })


    def _get_controlled_lanes(self, light_id: str) -> List[str]:
        if light_id not in self._controlled_lanes_cache:
            self._controlled_lanes_cache[light_id] = traci.trafficlight.getControlledLanes(light_id)
        return self._controlled_lanes_cache[light_id]

    def _get_downstream_lights(self, light_id: str) -> List[str]:
        if light_id not in self._downstream_lights_cache:
            downstream_lights = set()
            controlled_lanes = self._get_controlled_lanes(light_id)
            
            for lane_id in controlled_lanes:
                links = traci.lane.getLinks(lane_id)
                for link in links:
                    next_lane_id = link[0]
                    for downstream_light_id in self.light_ids:  # Only check lights we care about
                        if next_lane_id in self._get_controlled_lanes(downstream_light_id):
                            downstream_lights.add(downstream_light_id)
            
            self._downstream_lights_cache[light_id] = list(downstream_lights)
        return self._downstream_lights_cache[light_id]

    def _get_lane_metrics(self, lanes: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        queue_length = np.zeros(4, dtype=np.float32)
        waiting_time = np.zeros(4, dtype=np.float32)
        total_vehicles = 0

        for i, lane_id in enumerate(lanes[:4]):
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                halting = sum(1 for veh in vehicles if traci.vehicle.getSpeed(veh) < 0.1)
                queue_length[i] = halting
                waiting_time[i] = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
                total_vehicles += len(vehicles)
            except traci.exceptions.TraCIException:
                print(f"Error accessing lane metrics for lane: {lane_id}")
                continue

        # Prevent division by zero
        waiting_time = np.clip(waiting_time, 0, 1e3)
        return queue_length, waiting_time, total_vehicles

    def _get_observation(self, light_id: str) -> Dict:
        lanes = self._get_controlled_lanes(light_id)[:4]
        queue_length, waiting_time, _ = self._get_lane_metrics(lanes)
        
        vehicle_data = defaultdict(list)
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                speed = traci.vehicle.getSpeed(veh)
                vehicle_data['speeds'].append(float(speed))
                if speed < 0.1:
                    vehicle_data['idle'].append(1)
                vehicle_data['time_loss'].append(float(traci.vehicle.getTimeLoss(veh)))

        mean_speed = np.array([traci.lane.getLastStepMeanSpeed(lane) for lane in lanes], dtype=np.float32)
        idle_time = float(sum(vehicle_data['idle']))
        trip_penalty = float(np.mean(vehicle_data['time_loss']) / 60.0 if vehicle_data['time_loss'] else 0)
        
        downstream_lights = self._get_downstream_lights(light_id)
        downstream_queue = float(sum(np.sum(self._get_lane_metrics(self._get_controlled_lanes(dl))[0]) 
                            for dl in downstream_lights))
        
        current_time = traci.simulation.getTime()
        avg_switch_time = float(np.mean([
            traci.trafficlight.getNextSwitch(dl) - current_time
            for dl in downstream_lights
        ]) if downstream_lights else 0.0)

        return {
            "local_state": {
                "queue_length": np.clip(queue_length, 0, 120).astype(np.float32),
                "current_phase": np.array([traci.trafficlight.getPhase(light_id)], dtype=np.float32),
                "phase_state": self._get_phase_features(light_id).astype(np.float32),
                "waiting_time": np.clip(waiting_time / 100.0, 0, 1e3).astype(np.float32),
                "mean_speed": np.clip(mean_speed, 0, 30).astype(np.float32),
                "trip_penalty": np.clip(np.array([trip_penalty], dtype=np.float32), 0, 60),
                "idle_time": np.clip(np.array([idle_time], dtype=np.float32), 0, 1e3),
            },
            "downstream_context": {
                "total_downstream_queue": np.clip(np.array([downstream_queue], dtype=np.float32), 0, 400),
                "avg_time_to_switch": np.clip(np.array([avg_switch_time], dtype=np.float32), 0, 120),
            }
        }

    def _compute_reward(self, light_id: str) -> float:
        lanes = self._get_controlled_lanes(light_id)[:4]
        queue_length, waiting_time, throughput = self._get_lane_metrics(lanes)
        
        # Batch collect vehicle data
        speed_ratios = []
        idle_count = 0
        time_losses = []
        
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            if not vehicles:
                continue
                
            # Get max speed once and cache it
            if lane not in self._max_speed_cache:
                self._max_speed_cache[lane] = traci.lane.getMaxSpeed(lane)
            max_speed = self._max_speed_cache[lane]
            
            for veh in vehicles:
                speed = traci.vehicle.getSpeed(veh)
                if speed < 0.1:
                    idle_count += 1
                if max_speed > 0:
                    speed_ratios.append(speed / max_speed)
                time_losses.append(traci.vehicle.getTimeLoss(veh))

        speed_efficiency = np.mean(speed_ratios) if speed_ratios else 0
        trip_penalty = np.mean(time_losses) / 60.0 if time_losses else 0
        
        # Calculate downstream impacts
        downstream_lights = self._get_downstream_lights(light_id)
        downstream_queue = sum(
            np.sum(self._get_lane_metrics(self._get_controlled_lanes(dl))[0])
            for dl in downstream_lights
        )
        
        # Compute reward
        reward = (
            throughput * 1.5 +
            speed_efficiency * 2 -
            np.sum(queue_length) * 0.8 -
            np.sum(waiting_time) * 0.8 -
            idle_count * 1.8 -
            trip_penalty * 1.4 -
            downstream_queue * 0.3
        )
        
        return np.clip(reward, -10.0, 10.0)

    def _get_phase_features(self, light_id: str) -> np.ndarray:
        phase_state = traci.trafficlight.getRedYellowGreenState(light_id)
        features = np.zeros(4, dtype=np.float32)
        for i in range(min(4, len(phase_state))):
            if phase_state[i] == 'G': features[i] = 1.0
            elif phase_state[i] == 'y': features[i] = 0.5
        return features
    def _convert_observation_to_float32(self, obs: Dict) -> Dict:
        for light_id, observation in obs.items():
            observation["local_state"] = {
                k: v.astype(np.float32) if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for k, v in observation["local_state"].items()
            }
            observation["downstream_context"] = {
                k: v.astype(np.float32) if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
                for k, v in observation["downstream_context"].items()
            }
            obs[light_id] = observation
        return obs


    def reset(self, *, seed=None, options=None):
        self._start_simulation()
        self.current_step = 0
        
        for _ in range(10):
            traci.simulationStep()

        obs = {}
        for light_id in self.light_ids:
            observation = self._get_observation(light_id)
            obs[light_id] = observation

        # Convert observations to float32
        obs = self._convert_observation_to_float32(obs)
        
        # Validate observations
        for light_id, observation in obs.items():
            assert self.observation_space[light_id].contains(observation), f"Observation for {light_id} is invalid: {observation}"

        # Clear caches
        self._controlled_lanes_cache.clear()
        self._downstream_lights_cache.clear()
        self._lane_length_cache.clear()
        self._max_speed_cache.clear()

        self.light_phases = {
            light_id: len(traci.trafficlight.getAllProgramLogics(light_id)[0].phases)
            for light_id in self.light_ids
        }

        return obs, {}

    def step(self, action: Dict):
        self.current_step += 1
        
        # Batch process all light changes
        phase_changes = []
        for light_id, light_action in action.items():
            phase = int(light_action["phase"]) % self.light_phases[light_id]
            duration = float(light_action["duration"].item())
            duration = np.clip(duration, 5.0, 60.0)
            phase_changes.append((light_id, phase, duration))
        
        # Apply changes in batch
        for light_id, phase, duration in phase_changes:
            traci.trafficlight.setPhase(light_id, phase)
            traci.trafficlight.setPhaseDuration(light_id, duration)

        traci.simulationStep()

        # Collect observations and rewards efficiently
        obs = {light_id: self._get_observation(light_id) for light_id in self.light_ids}
        rewards = sum(self._compute_reward(light_id) for light_id in self.light_ids)
        
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.max_steps

        info = {
            "episode_step": self.current_step,
            "total_vehicles": traci.vehicle.getIDCount(),
            "simulation_time": traci.simulation.getTime()
        }

        return obs, rewards, terminated, truncated, info

    def _start_simulation(self):
        if traci.isLoaded():
            traci.close()
            
        retries = 3
        for attempt in range(retries):
            try:
                sumo_cmd = [
                    "sumo-gui" if self.config.get("gui", True) else "sumo",
                    "-c", self.sumo_cfg,
                    "--start",
                    "--quit-on-end", "true",
                    "--random",
                    "--time-to-teleport", "-1",
                    "--waiting-time-memory", "500",
                    "--no-warnings", "true",
                    "--step-length", "0.2",  
                    "--no-step-log",  
                    "--threads", str(self.config.get("num_threads", 4))  # Parallel execution
                ]
                traci.start(sumo_cmd)
                return
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to start SUMO after {retries} attempts: {e}")
                time.sleep(0.5)
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
        
        self.num_lights = len(obs_space.original_space.spaces)
        input_size = 24
        
        # Use Layer Normalization for better gradient flow
        self.input_norm = nn.LayerNorm(input_size)
        
        # Optimized encoder using residual connections
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),  # Additional layer
                nn.ReLU(),
            ) for _ in range(self.num_lights)
        ])
        
        combined_size = 16 * self.num_lights
        
        # Global feature processing with skip connections
        self.global_net = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # Additional layer
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
        # Value network with separate normalization
        self.value_net = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),  # Additional layer
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._value_out = None
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)

        # JIT compile forward pass for speed
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        device = self.device
        
        light_features = []
        for i, encoder in enumerate(self.encoder):
            light_id = list(obs.keys())[i]
            local_state = obs[light_id]["local_state"]
            downstream = obs[light_id]["downstream_context"]
            
            features = torch.cat([
                local_state["queue_length"].to(device),
                local_state["current_phase"].expand(-1, 4).to(device),
                local_state["phase_state"].to(device),
                local_state["waiting_time"].to(device),
                local_state["mean_speed"].to(device),
                local_state["trip_penalty"].to(device),
                local_state["idle_time"].to(device),
                downstream["total_downstream_queue"].to(device),
                downstream["avg_time_to_switch"].to(device)
            ], dim=-1)
            
            normalized = self.input_norm(features)
            encoded = encoder(normalized)
            light_features.append(encoded)
        
        combined = torch.cat(light_features, dim=-1)
        logits = self.global_net(combined)
        self._value_out = self.value_net(combined).squeeze(1)
        
        return logits, state
    
    def value_function(self):
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
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True  
    # Initialize Ray (connect to existing cluster)
    ray.init(address="auto")  # Automatically detects the cluster address


    config = get_single_agent_training_config()

    # Define the storage path where checkpoints are saved
    storage_path = r"\\WIN-Q5NH2TGKKGM\RLTraffic\ClusterResults"
    os.makedirs(storage_path, exist_ok=True)

    # Optionally, handle checkpoint restoration if needed
    latest_checkpoint = None
    # Example:
    # latest_checkpoint = "path/to/latest/checkpoint"

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
                stop={"training_iteration": 800},  # Define your stopping criteria
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=3,  # Save every 3 iterations
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

