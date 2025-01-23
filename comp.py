import os
import numpy as np
import traci
import time
from ray.rllib.algorithms.ppo import PPO
from single import TrafficLightEnv, TrafficLightModel, get_single_agent_training_config
from ray.rllib.models import ModelCatalog

def run_baseline():
    # Run normal SUMO simulation
    sumo_cmd = [
        "sumo-gui",
        "-c", r"C:\Users\pc\Documents\Trafic\data\sumo\simulation.sumocfg",
        "--start",
        "--quit-on-end", "true",
        "--random"
    ]
    
    traci.start(sumo_cmd)
    stats = {"waiting_time": [], "queue_length": [], "throughput": []}
    
    for _ in range(1000):  # 20 seconds @ 100ms steps
        traci.simulationStep()
        
        total_waiting = 0
        total_queue = 0
        total_throughput = 0
        
        for light_id in ["301", "520", "538", "257", "138"]:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            for lane in lanes:
                total_waiting += traci.lane.getWaitingTime(lane)
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_throughput += traci.lane.getLastStepVehicleNumber(lane)
        
        stats["waiting_time"].append(total_waiting)
        stats["queue_length"].append(total_queue)
        stats["throughput"].append(total_throughput)
        
    print("\n------------BASELINE RESULTS-------------------")
    print(f"Average waiting time: {np.mean(stats['waiting_time']):.2f}")
    print(f"Average queue length: {np.mean(stats['queue_length']):.2f}")
    print(f"Average throughput: {np.mean(stats['throughput']):.2f}")
    print("----------------------------------------------")

    traci.close()
    return stats

def run_rl_agent():
    ModelCatalog.register_custom_model("my_traffic_model", TrafficLightModel)
    checkpoint = r"C:\Users\pc\Documents\Trafic\results\PPO_2025-01-22_18-57-13\PPO_TrafficLightEnv_befd4_00000_0_2025-01-22_18-57-13\checkpoint_000060"
    trained_agent = PPO.from_checkpoint(checkpoint)
    env = TrafficLightEnv({
        "num_lights": 5,
        "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\simulation.sumocfg",
        "max_steps": 2000
    })
    
    stats = {"waiting_time": [], "queue_length": [], "throughput": []}
    obs, _ = env.reset()
    
    for _ in range(1000):
        action = trained_agent.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_waiting = sum(sum(obs[light_id]["local_state"]["waiting_time"]) 
                          for light_id in obs)
        total_queue = sum(sum(obs[light_id]["local_state"]["queue_length"]) 
                         for light_id in obs)
                         
        # Calculate throughput the same way as in training
        total_throughput = 0
        for light_id in obs:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            total_throughput += sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
        
        stats["waiting_time"].append(total_waiting)
        stats["queue_length"].append(total_queue)
        stats["throughput"].append(total_throughput)
        
        if terminated or truncated:
            break
            
    print("\n------------RL AGENT RESULTS-------------------")
    print(f"Average waiting time: {np.mean(stats['waiting_time']):.2f}")
    print(f"Average queue length: {np.mean(stats['queue_length']):.2f}")
    print(f"Average throughput: {np.mean(stats['throughput']):.2f}")
    print("----------------------------------------------")
    return stats

def compare():
    baseline_stats = run_baseline()
    rl_stats = run_rl_agent()
    
    print("\nComparison after simulation:")
    metrics = ["waiting_time", "queue_length", "throughput"]
    
    for metric in metrics:
        bl_avg = np.mean(baseline_stats[metric])
        rl_avg = np.mean(rl_stats[metric])
        
        # For throughput, higher is better, so flip the improvement calculation
        if metric == "throughput":
            improvement = ((rl_avg - bl_avg) / bl_avg) * 100
        else:
            improvement = ((bl_avg - rl_avg) / bl_avg) * 100
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"Baseline: {bl_avg:.2f}")
        print(f"RL Agent: {rl_avg:.2f}")
        print(f"Improvement: {improvement:.1f}%")
if __name__ == "__main__":
   compare()