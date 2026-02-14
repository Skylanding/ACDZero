import numpy as np


def cast_observations_to_float32(observations, agents):
    for agent_id in agents:
        if not isinstance(observations[agent_id], np.ndarray):
            observations[agent_id] = np.array(observations[agent_id], dtype=np.float32)
        else:
            observations[agent_id] = observations[agent_id].astype(np.float32)
    return observations

