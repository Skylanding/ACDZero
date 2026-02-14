from typing import Any, Dict, Optional

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from agents.centralized_observation_wrapper import CentralizedObservationWrapper


def create_enterprise_mae_env(steps: int = 100):
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )
    cyborg = CybORG(scenario_generator=scenario)
    return EnterpriseMAE(cyborg)


def create_cc4_env(env_config: Optional[Dict[str, Any]] = None):
    config = env_config or {}
    steps = config.get("steps", 100)
    return create_enterprise_mae_env(steps=steps)


def create_hybrid_env(
    use_deterministic_red: bool = True,
    use_centralized_obs: bool = True,
):
    red_agent_class = FiniteStateRedAgent if use_deterministic_red else FiniteStateRedAgent

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class,
    )
    cyborg = CybORG(scenario_generator=scenario)
    env = EnterpriseMAE(cyborg)
    if use_centralized_obs:
        env = CentralizedObservationWrapper(env)
    return env

