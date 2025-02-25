from theseus import AgentTheseus
from test import SimpleModel

agent = AgentTheseus.load("model_2502_1751")

print(agent.policy_network)
