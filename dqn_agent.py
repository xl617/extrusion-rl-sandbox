import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, Any
import logging

from models import DQN
from replay_memory import ReplayMemory, Transition
from config import (
    device, MODEL_CONFIG, TRAINING_CONFIG, TEMPERATURE_CONFIG, 
    FLOW_RATE_CONFIG, PATHS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNAgent:
    """Deep Q-Network agent for reinforcement learning.
    
    This agent implements the DQN algorithm with experience replay and target network updates.
    
    Args:
        state_size (int): Size of the state space
        action_size (int): Size of the action space
        action_size1 (int): Size of the secondary action space
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_size1: int,
        memory_capacity: int = TRAINING_CONFIG['memory_capacity'],
        batch_size: int = TRAINING_CONFIG['batch_size'],
        gamma: float = TRAINING_CONFIG['gamma'],
        epsilon_start: float = TRAINING_CONFIG['epsilon_start'],
        epsilon_end: float = TRAINING_CONFIG['epsilon_end'],
        epsilon_decay: float = TRAINING_CONFIG['epsilon_decay'],
        tao: float = TRAINING_CONFIG['tao'],
        lr: float = TRAINING_CONFIG['learning_rate'],
        steps_done: int = 0
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_size1 = action_size1
        self.interaction_counter = 0
        
        # Initialize replay memories
        self.memory = ReplayMemory(memory_capacity)
        self.memory1 = ReplayMemory(memory_capacity)

        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tao = tao
        self.lr = lr
        self.steps_done = steps_done

        # Initialize networks
        self.policy_net = DQN(state_size, action_size, action_size1).to(device)
        self.target_net = DQN(state_size, action_size, action_size1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.lr, 
            amsgrad=True
        )

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Selected actions for both action spaces
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * (self.steps_done / self.epsilon_decay)
        )
        self.steps_done += 1


        if sample > eps_threshold:
            #print("policy actions")
            #print("self.interaction_counter ", self.interaction_counter)
            with torch.no_grad():
                q_values, q_values1 = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
                if self.interaction_counter % TEMPERATURE_CONFIG['temp_action_interval'] != 0:
                    action1 = torch.tensor([[MODEL_CONFIG['action_range1']]], device=device)
                else:
                    action1 = q_values1.max(1)[1].view(1, 1)
        else:
            #print("random actions")
            action = torch.tensor(
                [[random.randint(0, MODEL_CONFIG['action_range'] * 2)]],
                device=device,
                dtype=torch.long
            )
            if self.interaction_counter % TEMPERATURE_CONFIG['temp_action_interval'] != 0:
                action1 = torch.tensor([[MODEL_CONFIG['action_range1']]], device=device)
            else:
                action1 = torch.tensor(
                    [[random.randint(0, MODEL_CONFIG['action_range1'] * 2)]],
                    device=device,
                    dtype=torch.long
                )

        return action, action1

    def optimize_model(self) -> Optional[float]:
        """Perform one step of optimization on the policy network.
        
        Returns:
            Optional[float]: Loss value if optimization was performed, None otherwise
        """
        if len(self.memory) < self.batch_size/2 or len(self.memory1) < self.batch_size/2:
            return None

        # Sample transitions from both memories
        transitions = (
            self.memory.sample(int(self.batch_size/2)) + 
            self.memory1.sample(int(self.batch_size/2))
        )

        # Convert batch of transitions to transition of batches
        batch = Transition(*zip(*transitions))

        # Compute mask of non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )

        # Compute Q(s_t, a) for all states
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        action1_batch = torch.cat(batch.action1)
        reward_batch = torch.cat(batch.reward)

        # Compute V(s_{t+1}) for all next states
        next_state_Q_values = torch.zeros(self.batch_size, device=device)
        next_state_Q_values1 = torch.zeros(self.batch_size, device=device)

        if any(non_final_mask):
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            with torch.no_grad():
                targetQ, targetQ1 = self.target_net(non_final_next_states)
                next_state_Q_values[non_final_mask] = targetQ.max(1)[0]
                next_state_Q_values1[non_final_mask] = targetQ1.max(1)[0]

        # Compute expected Q values
        expected_state_Q_values = (next_state_Q_values * self.gamma) + reward_batch
        expected_state_Q_values1 = (next_state_Q_values1 * self.gamma) + reward_batch

        # Compute Q(s_t, a)
        q_values, q_values1 = self.policy_net(state_batch)
        state_Q_values = q_values.gather(1, action_batch.unsqueeze(1))
        state_Q_values1 = q_values1.gather(1, action1_batch.unsqueeze(1))

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_Q_values, expected_state_Q_values.unsqueeze(1))
        loss1 = criterion(state_Q_values1, expected_state_Q_values1.unsqueeze(1))
        total_loss = loss + loss1

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return total_loss.item()

    def save_checkpoint(self, episode: int, reward_record: list) -> None:
        """Save model checkpoint.
        
        Args:
            episode (int): Current episode number
            reward_record (list): Record of rewards
        """
        # Create new checkpoints directory if it doesn't exist
        os.makedirs(PATHS['new_checkpoints_dir'], exist_ok=True)
        
        file_name = f"{episode}.pth"
        file_path = os.path.join(PATHS['new_checkpoints_dir'], file_name)
        
        checkpoint = {
            "episode": episode,
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "memory": self.memory.memory,
            "memory_pos": self.memory.position,
            "memory1": self.memory1.memory,
            "memory_pos1": self.memory1.position,
            "cumulative_reward_episode": reward_record
        }
        
        torch.save(checkpoint, file_path)
        logger.info(f"New checkpoint saved at episode {episode} in {file_path}")

    def load_checkpoint(self, episode: int) -> Tuple[int, list]:
        """Load model checkpoint from benchmark directory.
        
        Args:
            episode (int): Episode number to load
            
        Returns:
            Tuple[int, list]: Loaded episode number and reward record
        """
        # Create benchmark checkpoints directory if it doesn't exist
        os.makedirs(PATHS['benchmark_checkpoints_dir'], exist_ok=True)
        
        file_name = f"{episode}.pth"
        file_path = os.path.join(PATHS['benchmark_checkpoints_dir'], file_name)
        
        try:
            checkpoint = torch.load(file_path, map_location=device)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.memory.memory = checkpoint["memory"]
            self.memory.position = checkpoint["memory_pos"]
            self.memory1.memory = checkpoint["memory1"]
            self.memory1.position = checkpoint["memory_pos1"]
            reward_record = checkpoint["cumulative_reward_episode"]

            self.steps_done = episode * MODEL_CONFIG['max_transition']
            
            logger.info(f"Benchmark checkpoint loaded from episode {episode} from {file_path}")
            return episode, reward_record
            
        except FileNotFoundError:
            logger.warning(f"No benchmark checkpoint found for episode {episode} at {file_path}")
            return 0, [] 