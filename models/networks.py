import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from gymnasium.spaces import Dict, Discrete, Box

class CentralizedCriticRLModule(RLModule):
    def __init__(self, observation_space, action_space, **kwargs):
        """
        Initialize the centralized critic RL module.
        """
        super().__init__(**kwargs)
        self.num_lights = 5  # You can make this dynamic if needed
        self.action_space = action_space

        # Handle Dict action space
        if isinstance(action_space, Dict):
            # Create separate policy heads for each action dimension
            self.policy_heads = nn.ModuleDict()
            for key, space in action_space.items():
                if isinstance(space, Discrete):
                    self.policy_heads[key] = nn.Linear(256, space.n)
                elif isinstance(space, Box):
                    # For Box space, output mean and log_std
                    assert len(space.shape) == 1, "Only 1D Box spaces are supported"
                    self.policy_heads[key] = nn.Linear(256, 2 * space.shape[0])  # Mean and log_std
                else:
                    raise ValueError(f"Unsupported space type for key {key}: {type(space)}")
        else:
            raise ValueError(f"Expected Dict action space, got {type(action_space)}")

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(128, 256),  # Adjust input size if needed
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value head remains unchanged
        self.value_head = nn.Linear(256, 1)

    def _process_action_output(self, features):
        """
        Process network outputs into action distributions.
        """
        action_outputs = {}
        
        for key, head in self.policy_heads.items():
            space = self.action_space[key]
            if isinstance(space, Discrete):
                # For discrete actions, output logits directly
                action_outputs[key] = head(features)
            elif isinstance(space, Box):
                # For continuous actions, output mean and log_std
                output = head(features)
                action_dim = space.shape[0]
                mean, log_std = output[..., :action_dim], output[..., action_dim:]
                
                # Clamp log_std for numerical stability
                log_std = torch.clamp(log_std, -20, 2)
                
                # Scale mean to action space range
                scaled_mean = (space.high - space.low) * torch.sigmoid(mean) + space.low
                
                action_outputs[key] = torch.cat([scaled_mean, log_std], dim=-1)
        
        return action_outputs

    @override(RLModule)
    def forward_inference(self, inputs):
        """
        Compute the policy logits for inference.
        """
        x = self._process_inputs(inputs)
        features = self.shared_layers(x)
        return self._process_action_output(features)

    @override(RLModule)
    def forward_train(self, inputs):
        """
        Compute the policy logits and value for training.
        """
        x = self._process_inputs(inputs)
        features = self.shared_layers(x)
        policy_outputs = self._process_action_output(features)
        value = self.value_head(features)
        
        return {"policy_logits": policy_outputs, "value": value}

    def _process_inputs(self, inputs):
        # Combine all light observations into a single tensor
        obs_tensors = []
        for light_id in inputs.keys():
            obs = inputs[light_id]
            queue_length = obs["local_state"]["queue_length"]
            waiting_time = obs["local_state"]["waiting_time"]
            current_phase = F.one_hot(obs["local_state"]["current_phase"].long(), num_classes=3).float()

            vehicles_approaching = obs["upstream_info"]["vehicles_approaching"]
            time_to_arrival = obs["upstream_info"]["time_to_arrival"]
            upstream_phases = F.one_hot(obs["upstream_info"]["upstream_phases"].long(), num_classes=3).float()
            upstream_phases_flat = upstream_phases.view(upstream_phases.shape[0], -1)

            queue_lengths_ds = obs["downstream_info"]["queue_lengths"]
            phases_ds = F.one_hot(obs["downstream_info"]["phases"].long(), num_classes=3).float()
            phases_ds_flat = phases_ds.view(phases_ds.shape[0], -1)

            light_number_oh = F.one_hot(obs["position"]["light_number"].long(), num_classes=self.num_lights).float()
            is_first = obs["position"]["is_first"].float().unsqueeze(-1)
            is_last = obs["position"]["is_last"].float().unsqueeze(-1)

            # Concatenate features for this light
            obs_tensors.append(torch.cat([
                queue_length, waiting_time, current_phase, vehicles_approaching, time_to_arrival,
                upstream_phases_flat, queue_lengths_ds, phases_ds_flat, light_number_oh, is_first, is_last
            ], dim=-1))

        # Stack observations for all lights
        return torch.stack(obs_tensors, dim=0)


    @override(RLModule)
    def output_specs_inference(self):
        """
        Specify the output format for inference.
        """
        return {key: "tensor" for key in self.action_space.spaces.keys()}

    @override(RLModule)
    def output_specs_train(self):
        """
        Specify the output format for training.
        """
        return {
            "policy_logits": {key: "tensor" for key in self.action_space.spaces.keys()},
            "value": "tensor"
        }