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
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

# If you use traci, import it; else comment out
import traci

class TrafficLightMAEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        
        self.num_lights = config.get("num_lights", 5)
        self.light_ids = config.get(
            "light_ids", ["A0", "B0", "C0", "D0", "E0"][: self.num_lights]
        )
        self.sumo_cfg = config.get("sumo_cfg", "data/sumo/simulation.sumocfg")
        self.max_steps = config.get("max_steps", 10000)
        
        # Add a connection lock
        self._is_connected = False
        self.current_step = 0
        
        # Space definitions remain the same
        self.single_light_obs_space = gym.spaces.Dict({
                "local_state": gym.spaces.Dict({
                    "queue_length": gym.spaces.Box(
                        low=np.zeros(4, dtype=np.float32),
                        high=np.full(4, 100, dtype=np.float32),
                        dtype=np.float32
                    ),
                    "current_phase": gym.spaces.Discrete(3),
                    "waiting_time": gym.spaces.Box(
                        low=np.zeros(4, dtype=np.float32),
                        high=np.full(4, 1e6, dtype=np.float32),
                        dtype=np.float32
                    ),
                }),
                "neighbor_info": gym.spaces.Box(
                    low=np.zeros(4, dtype=np.float32),
                    high=np.full(4, 100, dtype=np.float32),
                    dtype=np.float32
                ),
            })        
        self.single_light_act_space = gym.spaces.Dict({
            "phase": gym.spaces.Discrete(3),
            "duration": gym.spaces.Box(low=5.0, high=60.0, shape=(1,), dtype=np.float32),
        })
        
        self.observation_space = self.single_light_obs_space
        self.action_space = self.single_light_act_space
        self.last_metrics = {lid: {
            "queue_length": 0.0,
            "waiting_time": 0.0,
            "throughput": 0.0
        } for lid in self.light_ids}

    def _ensure_connection(self):
        """Ensure SUMO connection is active"""
        if not self._is_connected:
            self._start_simulation()

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

        
    def reset(self, *, seed=None, options=None):
        """Reset environment with proper connection handling"""
        self._start_simulation()  # Always start fresh
        self.current_step = 0
        
        # Advance some simulation steps to fill initial data
        for _ in range(10):
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                print("Lost connection during reset, attempting to reconnect...")
                self._start_simulation()
                return self.reset(seed=seed, options=options)
        
        # Build multi-agent observation
        obs_dict = {}
        for lid in self.light_ids:
            obs_dict[lid] = self._get_observation(lid)
        
        return obs_dict, {}

    def step(self, action_dict):
        try:
            self._ensure_connection()
            self.current_step += 1
            
            # 1) Apply each agent's chosen phase/duration
            durations = []
            for lid, action in action_dict.items():
                phase = int(action["phase"])
                dur = float(action["duration"])
                dur = max(5.0, min(60.0, dur))
                
                try:
                    traci.trafficlight.setPhase(lid, phase)
                    traci.trafficlight.setPhaseDuration(lid, dur)
                    durations.append(dur)
                except traci.exceptions.TraCIException as e:
                    print(f"TraCI error for light {lid}: {e}")
                    self._is_connected = False
                    return self.reset()[0], {lid: 0.0 for lid in self.light_ids}, {"__all__": True}, {"__all__": False}, {}
            
            max_dur = int(max(durations))
            
            # 2) Accumulate rewards over sub-steps
            rewards_dict = {lid: 0.0 for lid in self.light_ids}
            for _ in range(max_dur):
                try:
                    traci.simulationStep()
                    for lid in self.light_ids:
                        rewards_dict[lid] += self._compute_reward(lid)
                except traci.exceptions.TraCIException:
                    self._is_connected = False
                    return self.reset()[0], rewards_dict, {"__all__": True}, {"__all__": False}, {}
            
            # 3) Build next observations and infos
            next_obs_dict = {}
            infos_dict = {}
            
            for lid in self.light_ids:
                try:
                    lanes = traci.trafficlight.getControlledLanes(lid)
                    queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes[:4])
                    waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes[:4])
                    throughput = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes[:4])
                    
                    self.last_metrics[lid] = {
                        "queue_length": float(queue_length),
                        "waiting_time": float(waiting_time),
                        "throughput": float(throughput)
                    }
                    next_obs_dict[lid] = self._get_observation(lid)
                    infos_dict[lid] = self.last_metrics[lid].copy()  # Use copy to avoid reference issues
                    
                except traci.exceptions.TraCIException as e:
                    print(f"Error collecting metrics for {lid}: {e}")
                    infos_dict[lid] = {}

            
            # 4) Global termination condition
            done = (
                traci.simulation.getMinExpectedNumber() <= 0
                or self.current_step >= self.max_steps
            )
            
            terminated_dict = {lid: done for lid in self.light_ids}
            truncated_dict = {lid: False for lid in self.light_ids}
            terminated_dict["__all__"] = done
            truncated_dict["__all__"] = False
            
            if done:
                try:
                    traci.close()
                    self._is_connected = False
                except Exception:
                    pass
            
            return next_obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict
            
        except Exception as e:
            print(f"Unexpected error in step: {str(e)}")
            self._is_connected = False
            return self.reset()[0], {lid: 0.0 for lid in self.light_ids}, {"__all__": True}, {"__all__": False}, {}

            

    # -------------------------
    # HELPER METHODS
    # -------------------------
    def _get_observation(self, light_id):
        """
        Build the observation for one traffic light `light_id`.
        We'll gather up to 4 lanes' queue lengths and waiting times,
        as well as the current 0..2 phase.
        """
        # 1) Gather queue length for up to 4 lanes
        queue_length = self._get_queue_length(light_id)

        # 2) Gather waiting time for up to 4 lanes
        waiting_time = self._get_waiting_time(light_id)

        # 3) Current phase (we clamp to 0..2 for G/Y/R)
        current_phase = 0
        try:
            phase = traci.trafficlight.getPhase(light_id)
            current_phase = phase % 3  # if your sumo has 3-phase logic
        except traci.exceptions.TraCIException:
            pass

        # 4) neighbor_info is placeholder
        neighbor_info = np.zeros(4, dtype=np.float32)


        obs = {
            "local_state": {
                "queue_length": queue_length,
                "current_phase": current_phase,
                "waiting_time": waiting_time,
            },
            "neighbor_info": neighbor_info,
        }
        return obs

    def _compute_reward(self, light_id):
        """Enhanced reward function"""
        # Queue length penalty
        queue_len_array = self._get_queue_length(light_id)
        queue_penalty = float(np.sum(queue_len_array))
        
        # Waiting time penalty
        waiting_time_array = self._get_waiting_time(light_id)
        waiting_penalty = float(np.sum(waiting_time_array)) / 100.0  # Scale down
        
        # Throughput reward (vehicles that passed)
        lanes = traci.trafficlight.getControlledLanes(light_id)
        throughput = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes[:4])
        
        # Combine rewards
        reward = (throughput * 2.0  # Positive reward for throughput
                - queue_penalty * 0.5  # Reduced penalty for queue
                - waiting_penalty * 0.3)  # Small penalty for waiting
        
        return reward

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

