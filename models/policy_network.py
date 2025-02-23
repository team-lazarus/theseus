import torch
import logging
from torch import nn
import torch.nn.functional as F

from theseus.utils import State


class PolicyDQN(nn.Module):


    def __init__(self, n_actions):
        super(PolicyDQN, self).__init__()
        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
        self.logger = logging.getLogger("PolicyDQN")

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def preprocess_state(self, state: State) -> torch.Tensor:
        self.logger.error("using default preprocess_state, rewrite preprocess_state for correct input")
 
        return torch.rand((1, 128))