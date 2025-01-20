import ray
from ray import tune
from ray.air import RunConfig
from ray.tune import Tuner
import traci
import sys

from config.trainingConfig import get_training_config
from environments.trafficEnv import TrafficLightEnv


def test_environment():
    env = None
    try:
        env = TrafficLightEnv()
        obs, _ = env.reset()

        print("Environment initialized successfully")
        print(f"Number of traffic lights: {len(env.light_ids)}")
        print(f"Traffic light IDs: {env.light_ids}")
        # Example of printing the first light's obs
        first_light = env.light_ids[0]
        print(f"Observation for first light:\n{obs[first_light]}")

        for step in range(10):
            print(f"\nStep {step + 1}")

            # Create actions dict
            actions = {
                light_id: {
                    "phase": 0,       # always 0 => e.g. "Green"
                    "duration": 30.0,
                }
                for light_id in env.light_ids
            }

            next_obs, rewards, dones, infos = env.step(actions)

            print(f"Rewards: {rewards}")
            print(f"Infos: {infos}")

            if dones["__all__"]:
                print("Simulation completed")
                break

        return env

    except Exception as e:
        print(f"Error occurred: {e}")
        if env:
            try:
                traci.close()
            except:
                pass
        raise


def train():
    ray.init()

    # Get the training config and run config
    config, run_config = get_training_config()

    # Use the Tuner API for training
    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=run_config,
    )

    # Run the training and get results
    results = tuner.fit()

    # Get the best checkpoint
    best_checkpoint = results.get_best_result().checkpoint
    print(f"Best checkpoint: {best_checkpoint}")


def main():
    MODE = "train"
    try:
        if MODE == "test":
            env = test_environment()
            if env:
                try:
                    print("\nRunning additional test steps ...")
                    for _ in range(5):
                        actions = {
                            light_id: {
                                "phase": 0,
                                "duration": 30.0
                            }
                            for light_id in env.light_ids
                        }
                        obs, rewards, dones, _ = env.step(actions)
                        if dones["__all__"]:
                            print("Simulation ended")
                            break
                finally:
                    traci.close()
                    print("TraCI connection closed.")
        else:
            train()

    except Exception as e:
        print(f"Fatal error: {e}")
        try:
            traci.close()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
