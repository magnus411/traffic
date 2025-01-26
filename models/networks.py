import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from gymnasium.spaces import  Discrete, Box
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian


from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from typing import Dict, Optional


class TrafficLightModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.num_lights = len(obs_space.original_space.spaces)
        input_size = 24
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),  
                nn.ReLU(),
            ) for _ in range(self.num_lights)
        ])
        
        combined_size = 16 * self.num_lights
        
        self.global_net = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
        self.value_net = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._value_out = None
        
        self.apply(self._init_weights)

    
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
    

