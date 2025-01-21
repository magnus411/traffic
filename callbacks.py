# callbacks.py

import wandb
import numpy as np
import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)

class TrafficLightCallbacks(DefaultCallbacks):
    def on_train_start(self, *, algorithm, **kwargs):
        # Initialize W&B at the start of training
        wandb.init(
            project="traffic-light-rl",
            config=algorithm.config,
            sync_tensorboard=True,  # Sync RLlib metrics to W&B
            reinit=True  # Allow multiple runs in the same script
        )
        logger.info("W&B training started.")

    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        **kwargs  # Removed 'env' and 'algorithm'
    ):
        # Initialize metrics storage at the start of each episode
        episode.user_data["queue_lengths"] = []
        episode.user_data["waiting_times"] = []
        episode.user_data["throughput"] = []
        logger.debug(f"Episode {episode.episode_id} started.")

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """
        Callback triggered at each step of an episode.
        Collect metrics for each agent.
        """
        # Iterate over all agents in the episode
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info:
                # Collect metrics
                queue_length = info.get("queue_length", 0)
                waiting_time = info.get("waiting_time", 0)
                throughput = info.get("throughput", 0)

                # Append to episode's user_data
                episode.user_data["queue_lengths"].append(queue_length)
                episode.user_data["waiting_times"].append(waiting_time)
                episode.user_data["throughput"].append(throughput)

                logger.debug(f"Episode {episode.episode_id} Agent {agent_id} metrics collected.")

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """
        Callback triggered at the end of an episode.
        Aggregate metrics and log.
        """
        # Calculate average and total metrics at episode end
        if episode.user_data["queue_lengths"]:
            avg_queue_length = np.mean(episode.user_data["queue_lengths"])
        else:
            avg_queue_length = 0.0

        if episode.user_data["waiting_times"]:
            avg_waiting_time = np.mean(episode.user_data["waiting_times"])
        else:
            avg_waiting_time = 0.0

        if episode.user_data["throughput"]:
            total_throughput = np.sum(episode.user_data["throughput"])
        else:
            total_throughput = 0.0

        # Log custom metrics to RLlib
        episode.custom_metrics["avg_queue_length"] = avg_queue_length
        episode.custom_metrics["avg_waiting_time"] = avg_waiting_time
        episode.custom_metrics["total_throughput"] = total_throughput

        # Log custom metrics to W&B
        wandb.log({
            "avg_queue_length": avg_queue_length,
            "avg_waiting_time": avg_waiting_time,
            "total_throughput": total_throughput,
            "episode_reward": episode.total_reward,  # Optional: log total reward
            "episode_length": episode.length  # Optional: log episode length
        })

        logger.debug(f"Episode {episode.episode_id} ended and metrics logged.")

    def on_train_end(self, *, algorithm, **kwargs):
        # Finish the W&B run at the end of training
        wandb.finish()
        logger.info("W&B training finished.")
