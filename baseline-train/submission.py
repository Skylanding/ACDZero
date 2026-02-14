import os

from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv

### Import custom agents here ###
from models.cage4 import load
from wrapper.graph_wrapper import GraphWrapper

class Submission:

    # Submission name
    NAME: str = "KEEP"

    # Name of your team
    TEAM: str = "Cybermonic"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Graph-based PPO With Intra-agent Communication"

    # Use this function to define your agents.
    AGENTS = {}
    for i in range(5):
        # Load checkpoint (adjust name if needed: _checkpoint.pt or _40k.pt etc.)
        ckpt_path = f'{os.path.dirname(__file__)}/checkpoints/run_mcts-{i}_checkpoint.pt'
        agent = load(ckpt_path)
        # Disable MCTS for eval: use pure actor policy (greedy)
        agent.planner = None
        agent.use_mcts = False
        agent.set_deterministic(True)
        AGENTS[f"blue_agent_{i}"] = agent

    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return GraphWrapper(env)