import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple

class DQN(nn.Module):
    """Deep Q-Network model for reinforcement learning.
    
    This network takes state observations and outputs Q-values for each possible action.
    It uses separate processing paths for video features and indicator values.
    
    Args:
        n_observations (int): Size of the state space
        n_actions (int): Size of the action space
        n_actions1 (int): Size of the secondary action space
    """
    
    def __init__(self, n_observations: int, n_actions: int, n_actions1: int):
        super(DQN, self).__init__()
        
        # Determine the size of video features and indicator based on n_observations
        n_video_features = n_observations - 2  # Assuming last 2 elements are indicators
        n_indicator = 2

        # Video features processing path
        self.fc1_video = nn.Linear(n_video_features, 256)
        self.relu1_video = nn.ReLU()

        # Indicator processing path
        self.fc1_indicator = nn.Linear(n_indicator, 32)
        self.relu1_indicator = nn.ReLU()

        # Combined processing
        self.fc_merged = nn.Linear(256 + 32, 128)
        self.relu_merged = nn.ReLU()

        # Output layers for both action spaces
        self.fc_output = nn.Linear(128, n_actions)
        self.fc_output1 = nn.Linear(128, n_actions1)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        """Initialize network weights using He initialization.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q-values for both action spaces
        """
        # Separate the video features and indicator value from the input
        video_input = x[:, :-2]  # All columns except the last two
        indicator_input = x[:, -2:]  # Last two columns

        # Process video features
        x_video = self.relu1_video(self.fc1_video(video_input))
        
        # Process indicator values
        x_indicator = self.relu1_indicator(self.fc1_indicator(indicator_input))

        # Combine the processed features
        x_merged = torch.cat((x_video, x_indicator), dim=1)
        x_merged = self.relu_merged(self.fc_merged(x_merged))

        # Get Q-values for both action spaces
        q_values = self.fc_output(x_merged)
        q_values1 = self.fc_output1(x_merged)
        
        return q_values, q_values1 