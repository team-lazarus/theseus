import random
from collections import deque
from typing import List, Tuple

class ExperienceReplayMemory(object):
    def __init__(self, max_length: int, seed: int=42):
        self.memory = deque([], maxlen=max_length)

        self.seed = seed
        random.seed(seed)
    
    def append(self, transition: Tuple):
        self.memory.append(transition)
    
    def sample(self, sample_size:int) -> List[int]:
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
