import ray
from ray.rllib.algorithms.ppo import PPO
from environments.trafficEnv import TrafficLightMAEnv
import traci
from ray.rllib.models import ModelCatalog  # Add this import
from models.networks import MyTrafficModel  
def evaluate_model():
    # Initialize ray
    ray.init(ignore_reinit_error=True)
    
    # Register the custom model
    ModelCatalog.register_custom_model("my_traffic_model", MyTrafficModel)

    # Load the trained checkpoint
    checkpoint_path = "C:/Users/pc/Documents/Trafic/results/PPO_2025-01-21_09-10-43/PPO_TrafficLightMAEnv_a5bb6_00000_0_2025-01-21_09-10-43/checkpoint_000005"
    trained_model = PPO.from_checkpoint(checkpoint_path)

    # Create environment with GUI enabled
    env = TrafficLightMAEnv({
        "num_lights": 5,
        "sumo_cfg": r"C:\Users\pc\Documents\Trafic\data\sumo\simulation.sumocfg",
        "max_steps": 3000
    })

    # Run evaluation loop
    episode_reward = 0
    obs, info = env.reset()  # Modified to unpack properly
    done = {"__all__": False}
    step = 0

    while not done["__all__"]:
        # Compute actions for each agent
        actions = {}
        for agent_id, agent_obs in obs.items():  # Modified to handle dict properly
            actions[agent_id] = trained_model.compute_single_action(
                observation=agent_obs,  # Pass the observation directly
                policy_id="shared_policy"
            )

        # Execute actions
        obs, rewards, dones, truncated, info = env.step(actions)  # Modified to match Gym API
        
        # Sum rewards
        episode_reward += sum(rewards.values())
        done = dones
        step += 1

        print(f"Step {step}, Reward: {sum(rewards.values())}")

    print(f"Episode finished with total reward: {episode_reward}")
    env.close()

if __name__ == "__main__":
    evaluate_model()

