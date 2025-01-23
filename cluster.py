
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.air.integrations.wandb import WandbLoggerCallback

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
        self.light_ids = config.get("light_ids", ["301", "520", "538", "257", "138"][: self.num_lights])
        self.sumo_cfg = config.get("sumo_cfg", "data/sumo/simulation.sumocfg")
        self.max_steps = config.get("max_steps", 10000)
        self._is_connected = False
        self.current_step = 0
        



        # We'll initialize this after starting SUMO
        self.light_phases = {}

        # Use a default max phases for initialization
        max_default_phases = 10  # Maximum possible phases any light might have
        
        # Combined observation space
        self.observation_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "local_state": gym.spaces.Dict({
                    "queue_length": gym.spaces.Box(low=0, high=120, shape=(4,), dtype=np.float32),
                    "current_phase": gym.spaces.Box(low=0, high=max_default_phases-1, shape=(1,), dtype=np.float32),
                    "phase_state": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                    "waiting_time": gym.spaces.Box(low=0, high=1e3, shape=(4,), dtype=np.float32),
                    "mean_speed": gym.spaces.Box(low=0, high=30, shape=(4,), dtype=np.float32),
                    "trip_penalty": gym.spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
                    "idle_time": gym.spaces.Box(low=0, high=1e3, shape=(1,), dtype=np.float32),
                }),
                "downstream_context": gym.spaces.Dict({
                    "total_downstream_queue": gym.spaces.Box(low=0, high=400, shape=(1,), dtype=np.float32),
                    "avg_time_to_switch": gym.spaces.Box(low=0, high=120, shape=(1,), dtype=np.float32),
                }),

            }) for light_id in self.light_ids
        })        
        self.action_space = gym.spaces.Dict({
            light_id: gym.spaces.Dict({
                "phase": gym.spaces.Discrete(max_default_phases),  # Use max possible phases
                "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
            }) for light_id in self.light_ids
        })



    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        self._start_simulation()
        self.current_step = 0

        # Now that SUMO is started, we can get the actual phase information
        self.light_phases = {
            light_id: len(traci.trafficlight.getAllProgramLogics(light_id)[0].phases)
            for light_id in self.light_ids
        }

        for _ in range(10):  # Warm-up steps
            traci.simulationStep()

        #TEMERAROY
        obs = {light_id: self._get_observation(light_id) for light_id in self.light_ids}
        info = {}
        
        return obs, info
    def step(self, action):
        """Execute one step of the environment."""
        self._ensure_connection()
        self.current_step += 1

        
            
        for light_id, light_action in action.items():


            max_phases = self.light_phases[light_id]

            phase = int(light_action["phase"]) % max_phases
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
    
    def _get_downstream_lights(self, light_id):
        downstream_lights = set()  # Use a set to avoid duplicates
        controlled_lanes = traci.trafficlight.getControlledLanes(light_id)  # Get lanes controlled by this light

        for lane_id in controlled_lanes:
            # Get links to downstream lanes
            links = traci.lane.getLinks(lane_id)  # Returns tuples of (next_lane_id, ...)
            for link in links:
                next_lane_id = link[0]  # The ID of the next lane
                # Find the traffic light controlling the downstream lane
                for downstream_light_id in traci.trafficlight.getIDList():
                    if next_lane_id in traci.trafficlight.getControlledLanes(downstream_light_id):
                        downstream_lights.add(downstream_light_id)

        return list(downstream_lights)

    def _get_observation(self, light_id):
        """Get the observation for a specific traffic light."""
        lanes = traci.trafficlight.getControlledLanes(light_id)
        
        # Get queue length for each lane (used in reward)
        queue_length = self._get_queue_length(light_id)  # Shape: [4]
        queue_length = np.clip(queue_length, 0, 120).astype(np.float32)

        # Get waiting time normalized to 0-1 range (used in reward)
        waiting_time = self._get_waiting_time(light_id) / 100.0
        waiting_time = np.clip(waiting_time, 0, 1e3)  # Prevent extreme values

        # Get current phase number
        current_phase = np.array([traci.trafficlight.getPhase(light_id)], dtype=np.float32)
        
        # Get phase state (GGrr etc.) and convert to numeric features
        phase_state = traci.trafficlight.getRedYellowGreenState(light_id)
        phase_features = np.zeros(4, dtype=np.float32)
        controlled_links = traci.trafficlight.getControlledLinks(light_id)
        for i in range(min(4, len(controlled_links))):
            if i < len(phase_state):
                if phase_state[i] == 'G': 
                    phase_features[i] = 1.0  # Green
                elif phase_state[i] == 'y': 
                    phase_features[i] = 0.5  # Yellow
                elif phase_state[i] == 'r': 
                    phase_features[i] = 0.0  # Red

        # Initialize lane density, mean speed, and idle time
        lane_density = np.zeros(4, dtype=np.float32)
        mean_speed = np.zeros(4, dtype=np.float32)
        idle_time = np.zeros(4, dtype=np.float32)
        # Collect lane-level metrics
        for i in range(min(4, len(lanes))):
            lane_id = lanes[i]
            try:
                # Get density (vehicles per meter)
                lane_length = traci.lane.getLength(lane_id)
                num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                lane_density[i] = num_vehicles / lane_length if lane_length > 0 else 0
                
                # Get mean speed (m/s)
                mean_speed[i] = traci.lane.getLastStepMeanSpeed(lane_id)

                # Calculate idle time (vehicles moving below a threshold speed)
                idle_time[i] = sum(
                    1.0
                    for veh in traci.lane.getLastStepVehicleIDs(lane_id)
                    if traci.vehicle.getSpeed(veh) < 0.1
                )
            except traci.exceptions.TraCIException:
                pass
        idle_time = np.clip(idle_time, 0, 1e3)  
        # Calculate additional features for trip times and trip penalty
        trip_times = [
            traci.vehicle.getTimeLoss(veh)
            for lane in lanes
            for veh in traci.lane.getLastStepVehicleIDs(lane)
        ]
        avg_trip_time = np.mean(trip_times) if trip_times else 0  # Average trip time
        trip_penalty = avg_trip_time / 60.0  # Normalize to minutes
        trip_penalty = np.clip(trip_penalty, 0, 60)  # Clip to prevent extreme values
        # Add features as normalized values
        trip_penalty_feature = np.array([trip_penalty], dtype=np.float32)
        idle_time_feature = np.sum(idle_time)  # Aggregate idle time across all lanes

        downstream_lights = self._get_downstream_lights(light_id)
        total_downstream_queue = sum(np.sum(self._get_queue_length(downstream)) for downstream in downstream_lights)
        total_downstream_queue = np.array([total_downstream_queue], dtype=np.float32)  # Shape: [1]
        total_downstream_queue = np.clip(total_downstream_queue, 0, 400).astype(np.float32)



        avg_time_to_switch = np.mean([
            traci.trafficlight.getNextSwitch(downstream) - traci.simulation.getTime()
            for downstream in downstream_lights
        ]) if downstream_lights else 0.0
        avg_time_to_switch = np.array([avg_time_to_switch], dtype=np.float32)  # Shape: [1]
        avg_time_to_switch = np.clip(avg_time_to_switch, 0, 120).astype(np.float32)


        # Combine all observations into a dictionary
        return {
            "local_state": {
                "queue_length": queue_length,           # Queue lengths per lane
                "current_phase": current_phase,         # Current phase
                "phase_state": phase_features,          # Red-Yellow-Green state features
                "waiting_time": waiting_time,           # Waiting time per lane
                "mean_speed": mean_speed,               # Mean speed per lane
                "trip_penalty": trip_penalty_feature,   # Trip penalty based on trip time
                "idle_time": np.array([idle_time_feature], dtype=np.float32),  # Total idle time

            },
            "downstream_context": {
                "total_downstream_queue": total_downstream_queue,
                "avg_time_to_switch": avg_time_to_switch,
            },


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





        trip_time_penalty = sum(
            traci.vehicle.getTimeLoss(veh)
            for lane in lanes
            for veh in traci.lane.getLastStepVehicleIDs(lane)
        )
        trip_time_penalty = np.clip(trip_time_penalty / 100.0, 0, 10)



        idle_time = sum(
            1.0 for lane in lanes 
            for veh in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getSpeed(veh) < 0.1
        )     
        trip_times = [
            traci.vehicle.getTimeLoss(veh)
            for lane in lanes
            for veh in traci.lane.getLastStepVehicleIDs(lane)
        ]
        avg_trip_time = np.mean(trip_times) if trip_times else 0  
        trip_penalty = (avg_trip_time / 60.0) 



        speed_scores = []
        for lane in lanes:
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            if max_speed > 0:
                speed_scores.append(mean_speed / max_speed)
        speed_efficiency = np.mean(speed_scores) if speed_scores else 0
        
        # Density penalty: penalize high density which indicates congestion

        # Emergency stop penalty: penalize emergency stops which indicate poor timing
        emergency_stops = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        

        downstream_lights = self._get_downstream_lights(light_id)
        downstream_queue_penalty = sum(np.sum(self._get_queue_length(downstream)) for downstream in downstream_lights)
        synchronization_penalty = sum(
            abs(traci.trafficlight.getPhase(light_id) - traci.trafficlight.getPhase(downstream))
            for downstream in downstream_lights
        )


        reward = (
                throughput * 1.5
                + speed_efficiency * 2
                - queue_length * 0.8
                - waiting_time * 0.8
                - idle_time * 1.8
                - trip_penalty * 1.4     
                - emergency_stops * 1
                - downstream_queue_penalty * 0.3
                - synchronization_penalty * 0.3

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
                    "sumo-gui",
                    "-c", self.sumo_cfg,
                    "--start",
                    "--quit-on-end", "true",
                    "--random",
                    "--time-to-teleport", "-1",
                    "--waiting-time-memory", "500",
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
        
        # Input size is 24: 6 features × 4 values each
        input_size = 24  # queue_length(4) + current_phase(4) + phase_state(4) + waiting_time(4) + 
                        #  + mean_speed(4) +  + trip_penalty(1) + idle_time(1)
        
        # Define sub-model for processing each traffic light's local state
        self.light_state_encoder = nn.Sequential(
            nn.LayerNorm(input_size),  # Normalize the 24 input features
            nn.Linear(input_size, 64),  # First layer from 24->64 (not 20->64)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Calculate combined feature size
        combined_size = 32 * self.num_lights
        
        # Global feature combiner
        self.global_combiner = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )
        
        # Separate value branch
        self.value_branch = nn.Sequential(
            nn.LayerNorm(combined_size),
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
            
            # Convert all observations to tensors
            queue_length = torch.as_tensor(local_state["queue_length"], device=self.device).float()
            waiting_time = torch.as_tensor(local_state["waiting_time"], device=self.device).float()
            current_phase = torch.as_tensor(local_state["current_phase"], device=self.device).float()
            phase_state = torch.as_tensor(local_state["phase_state"], device=self.device).float()
            mean_speed = torch.as_tensor(local_state["mean_speed"], device=self.device).float()
            trip_penalty = torch.as_tensor(local_state["trip_penalty"], device=self.device).float()
            idle_time = torch.as_tensor(local_state["idle_time"], device=self.device).float()
            
            downstream_context = obs[light_id]["downstream_context"]
            total_downstream_queue = torch.as_tensor(downstream_context["total_downstream_queue"], device=self.device).float()
            avg_time_to_switch = torch.as_tensor(downstream_context["avg_time_to_switch"], device=self.device).float()



            # Ensure current_phase is the right shape before expanding
            if current_phase.dim() == 1:
                current_phase = current_phase.unsqueeze(-1)
            
            # Expand current_phase from [batch_size, 1] to [batch_size, 4]
            current_phase = current_phase.expand(-1, 4)
            
            # Concatenate all features
            light_obs = torch.cat([
                queue_length,  # Shape: (4,)
                current_phase,  # Shape: (1,)
                phase_state,  # Shape: (4,)
                waiting_time,  # Shape: (4,)
                mean_speed,  # Shape: (4,)
                trip_penalty,  # Shape: (1,)
                idle_time,  # Shape: (1,)
                total_downstream_queue,  # Shape: (1,)
                avg_time_to_switch,  # Shape: (1,)
            ], dim=-1)
                


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
                "max_steps": 18000,
            },
        )
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2,
            rollout_fragment_length=200,
            sample_timeout_s=320,
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


from ray.air import RunConfig, CheckpointConfig  # Added CheckpointConfig import




def train():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Initialize Ray (connect to existing cluster)
    #ray.init(address='auto')  # Automatically detects the cluster address
    ray.init()  # Automatically detects the cluster address

    print("Connected to Ray cluster.")
    print("Available GPU IDs:", ray.get_gpu_ids())
    print("Available resources:", ray.available_resources())

    config = get_single_agent_training_config()

    # Define the storage path where checkpoints are saved
    storage_path = os.path.abspath("./results")
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
                stop={"training_iteration": 600},  # Define your stopping criteria
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


if __name__ == "__main__":
    train()
