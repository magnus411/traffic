import numpy as np
import traci
import os
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from environments.trafficEnv import TrafficLightEnv
from models.networks import TrafficLightModel

import ray

def run_baseline():
    sumo_cmd = [
        "sumo",
        "-c", r"C:\Users\pc\Documents\Trafic\data\sumo\Sim03\osm.sumocfg",
        "--start",
        "--quit-on-end", "true",
        "--random"
    ]
    traci.start(sumo_cmd)
    stats = {"waiting_time": [], "queue_length": [], "throughput": [], "idling_time": [], "trip_time": []}

    for _ in range(4000):  # Simulate 20 seconds @ 100ms steps
        traci.simulationStep()
        stats = collect_metrics(stats)
    traci.close()

    print_results("BASELINE RESULTS", stats)
    return stats


def run_rl_agent():
    ModelCatalog.register_custom_model("my_traffic_model", TrafficLightModel)
    checkpoint = r"C:\Users\pc\Documents\Trafic\newRes\PPO_2025-01-24_03-51-14\PPO_TrafficLightEnv_141fd_00000_0_2025-01-24_03-51-15\checkpoint_000109"
    trained_agent = PPO.from_checkpoint(checkpoint)
    env = TrafficLightEnv({
        "num_lights": 5,
        "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\Sim03\osm.sumocfg",
        "max_steps": 10000
    })

    stats = {"waiting_time": [], "queue_length": [], "throughput": [], "idling_time": [], "trip_time": []}
    obs, _ = env.reset()

    for _ in range(4000):
        action = trained_agent.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = collect_metrics(stats, obs)
        if terminated or truncated:
            break

    print_results("RL AGENT RESULTS", stats)
    return stats


def collect_metrics(stats, obs=None):
    total_waiting = 0
    total_queue = 0
    total_throughput = 0
    total_idling_time = 0
    total_trip_time = []

    if obs:
        for light_id in obs:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            total_waiting += sum(obs[light_id]["local_state"]["waiting_time"])
            total_queue += sum(obs[light_id]["local_state"]["queue_length"])
            total_throughput += sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)

            for lane in lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                total_idling_time += sum(
                    traci.vehicle.getAccumulatedWaitingTime(veh) for veh in vehicles
                )
                total_trip_time.extend(
                    [traci.vehicle.getTimeLoss(veh) for veh in vehicles]
                )
    else:
        for light_id in [
            "cluster_155251477_4254141700_6744555967_78275",
            "cluster_1035391877_6451996796",
            "cluster_163115359_2333391359_2333391361_78263_#1more",
            "cluster_105371_4425738061_4425738065_4425738069",
            "cluster_105372_4916088731_4916088738_4916088744"]:
            lanes = traci.trafficlight.getControlledLanes(light_id)
            for lane in lanes:
                total_waiting += traci.lane.getWaitingTime(lane)
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_throughput += traci.lane.getLastStepVehicleNumber(lane)

                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                total_idling_time += sum(
                    traci.vehicle.getAccumulatedWaitingTime(veh) for veh in vehicles
                )
                total_trip_time.extend(
                    [traci.vehicle.getTimeLoss(veh) for veh in vehicles]
                )

    stats["waiting_time"].append(total_waiting)
    stats["queue_length"].append(total_queue)
    stats["throughput"].append(total_throughput)
    stats["idling_time"].append(total_idling_time)
    if total_trip_time:
        stats["trip_time"].append(np.mean(total_trip_time))

    return stats


def print_results(title, stats):
    print(f"\n------------{title}-------------------")
    print(f"Average waiting time: {np.mean(stats['waiting_time']):.2f}")
    print(f"Average queue length: {np.mean(stats['queue_length']):.2f}")
    print(f"Average throughput: {np.mean(stats['throughput']):.2f}")
    print(f"Average idling time: {np.mean(stats['idling_time']):.2f}")
    if stats["trip_time"]:
        print(f"Average trip time: {np.mean(stats['trip_time']):.2f}")
    print("----------------------------------------------")


def plot_comparison(baseline_stats, rl_stats):
    metrics = ["waiting_time", "queue_length", "throughput", "idling_time", "trip_time"]
    for metric in metrics:
        if metric not in baseline_stats or not baseline_stats[metric]:
            continue
        plt.figure()
        plt.plot(baseline_stats[metric], label="Baseline")
        plt.plot(rl_stats[metric], label="RL Agent")
        plt.xlabel("Simulation Steps")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Over Time")
        plt.legend()
        plt.savefig(f"{metric}_comparison.png")
        plt.show()

    avg_baseline = [np.mean(baseline_stats[metric]) for metric in metrics if metric in baseline_stats and baseline_stats[metric]]
    avg_rl = [np.mean(rl_stats[metric]) for metric in metrics if metric in rl_stats and rl_stats[metric]]

    x = np.arange(len(avg_baseline))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, avg_baseline, width, label="Baseline")
    plt.bar(x + width / 2, avg_rl, width, label="RL Agent")
    plt.xlabel("Metrics")
    plt.ylabel("Average Value")
    plt.xticks(x, [metric.replace("_", " ").title() for metric in metrics if metric in baseline_stats and baseline_stats[metric]])
    plt.title("Average Metrics Comparison")
    plt.legend()
    plt.savefig("average_metrics_comparison.png")
    plt.show()


def compare():
    ray.init(local_mode=True, num_cpus=2)

    baseline_stats = run_baseline()
    rl_stats = run_rl_agent()

    print("\nComparison after simulation:")
    metrics = ["waiting_time", "queue_length", "throughput", "idling_time", "trip_time"]

    for metric in metrics:
        if metric not in baseline_stats or not baseline_stats[metric]:
            continue
        bl_avg = np.mean(baseline_stats[metric])
        rl_avg = np.mean(rl_stats[metric])

        if metric == "throughput":
            improvement = ((rl_avg - bl_avg) / bl_avg) * 100
        else:
            improvement = ((bl_avg - rl_avg) / bl_avg) * 100

        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"Baseline: {bl_avg:.2f}")
        print(f"RL Agent: {rl_avg:.2f}")
        print(f"Improvement: {improvement:.1f}%")

        t_stat, p_val = ttest_ind(baseline_stats[metric], rl_stats[metric])
        print(f"T-Test: t-stat={t_stat:.2f}, p-value={p_val:.4f}")

    plot_comparison(baseline_stats, rl_stats)


if __name__ == "__main__":
    compare()
