from ray.rllib.algorithms.ppo import PPOConfig
from ray.air import RunConfig
import os
from environments.trafficEnv import TrafficLightEnv
from models.networks import CentralizedCriticRLModule
from ray.rllib.core.rl_module import RLModuleSpec

def get_training_config():
    config = (
        PPOConfig()
        .environment(
            env=TrafficLightEnv,
            env_config={
                "num_lights": 5,
                "sumo_cfg": "data/sumo/simulation.sumocfg",
                "max_steps": 10000,
            }
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CentralizedCriticRLModule,
                observation_space=TrafficLightEnv.observation_space,
                action_space=TrafficLightEnv.action_space,
            )
        )
        .env_runners(
            num_env_runners=4,  # Number of environment runners
            num_envs_per_env_runner=1,  # Number of environments per runner
            num_cpus_per_env_runner=1,  # Number of CPUs per environment runner
            rollout_fragment_length=200,  # Number of steps per rollout fragment
            batch_mode="truncate_episodes",  # Rollout batch mode
        )
        .training(
            train_batch_size=4000,
            gamma=0.99,
            kl_coeff=0.3,
            vf_loss_coeff=1.0,
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,
        )
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    storage_path = os.path.abspath("./results")
    run_config = RunConfig(
        storage_path=storage_path,
        stop={"training_iteration": 100},
    )

    return config, run_config
