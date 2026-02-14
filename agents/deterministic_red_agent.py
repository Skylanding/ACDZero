
import numpy as np
from typing import Dict, Any, Optional


class DeterministicRedAgent:

    def __init__(self, agent_id: str = "red_agent_0"):
        self.agent_id = agent_id


        self.phase_duration = {
            'discovery': 20,
            'exploitation': 30,
            'escalation': 30,
            'impact': 20
        }


        self.target_priorities = [
            'contractor',
            'restricted_a',
            'operational_a',
            'restricted_b',
            'operational_b',
        ]


        self.current_phase = 'discovery'


        self.step_count = 0


        self.target_index = 0


        self.attacked_targets = set()


        self.phase_step_count = 0

    def get_action(self, observation: np.ndarray, action_space: Any) -> int:
        self.step_count += 1
        self.phase_step_count += 1


        self._update_phase()


        num_actions = action_space.n if hasattr(action_space, 'n') else len(action_space)

        if self.current_phase == 'discovery':
            action = self._discovery_action(num_actions)
        elif self.current_phase == 'exploitation':
            action = self._exploitation_action(num_actions)
        elif self.current_phase == 'escalation':
            action = self._escalation_action(num_actions)
        else:
            action = self._impact_action(num_actions)

        return action

    def _update_phase(self):
        total_phase_steps = sum(self.phase_duration.values())
        cycle_position = self.step_count % total_phase_steps

        if cycle_position < self.phase_duration['discovery']:
            new_phase = 'discovery'
        elif cycle_position < self.phase_duration['discovery'] + self.phase_duration['exploitation']:
            new_phase = 'exploitation'
        elif cycle_position < self.phase_duration['discovery'] + self.phase_duration['exploitation'] + self.phase_duration['escalation']:
            new_phase = 'escalation'
        else:
            new_phase = 'impact'


        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.phase_step_count = 0

    def _discovery_action(self, num_actions: int) -> int:


        scan_actions = min(5, num_actions)
        action_index = self.phase_step_count % scan_actions
        return action_index

    def _exploitation_action(self, num_actions: int) -> int:

        if num_actions > 15:
            attack_start = 5
            attack_range = 10

            target_offset = self.target_index % len(self.target_priorities)
            action_index = attack_start + (self.phase_step_count % attack_range)
            return action_index
        else:

            return (self.phase_step_count * 2) % num_actions

    def _escalation_action(self, num_actions: int) -> int:

        if num_actions > 25:
            escalation_start = 15
            escalation_range = 10
            action_index = escalation_start + (self.phase_step_count % escalation_range)
            return action_index
        else:
            return (self.phase_step_count * 3) % num_actions

    def _impact_action(self, num_actions: int) -> int:

        if num_actions > 35:
            impact_start = 25
            impact_range = 10
            action_index = impact_start + (self.phase_step_count % impact_range)
            return action_index
        else:
            return (self.phase_step_count * 4) % num_actions

    def reset(self):
        self.current_phase = 'discovery'
        self.step_count = 0
        self.target_index = 0
        self.attacked_targets = set()
        self.phase_step_count = 0


class DeterministicRedAgentWrapper:

    def __init__(self, agent_id: str = "red_agent_0"):
        self.agent = DeterministicRedAgent(agent_id)

    def get_action(self, observation: np.ndarray, action_space: Any) -> int:
        return self.agent.get_action(observation, action_space)

    def reset(self):
        self.agent.reset()
