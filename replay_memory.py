import random
from collections import namedtuple
from typing import List, Tuple

Transition = namedtuple(
    "Transition", ("state", "action", "action1", "next_state", "reward", "done")
)

class ReplayMemory:
    """Replay memory for storing and sampling transitions.
    
    This class implements experience replay, which is a technique used in DQN to break
    the correlation between consecutive samples and improve training stability.
    
    Args:
        capacity (int): Maximum number of transitions to store
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, *args) -> None:
        """Store a transition in the replay memory.
        
        If the memory is at full capacity, it overwrites the oldest transition.
        
        Args:
            *args: Components of a transition (state, action, action1, next_state, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions from the replay memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            List[Transition]: List of sampled transitions
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the current size of the replay memory."""
        return len(self.memory) 