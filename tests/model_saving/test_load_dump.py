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


def test_agent_dump():
    model = SimpleModel(1, 1)
    agent = AgentTheseus(model, model)
    result = agent.dump()
    assert result is not None


def test_agent_load():
    agent = AgentTheseus.load()
    assert agent is not None
    assert agent.policy_network is not None
    assert agent.policy_network is not None
