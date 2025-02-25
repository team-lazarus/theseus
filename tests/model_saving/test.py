from theseus import AgentTheseus
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel(1, 1)

agent = AgentTheseus(model, model)

agent.dump()
