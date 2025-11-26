# ==============================================================================
# FILE: cmdp_spma/helpers.py 
# build_reward_table, build_uniform_cost_table
# ==============================================================================
import numpy as np

def build_reward_table(env):
    """
    For tabular gymnasium envs with env.unwrapped.P (like FrozenLake),
    build the one-step expected reward table r(s,a).
    """
    import gymnasium as gym
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    nS, nA = env.observation_space.n, env.action_space.n
    P = env.unwrapped.P
    r = np.zeros((nS, nA), dtype=np.float32)
    for s in range(nS):
        for a in range(nA):
            transitions = P[s][a]
            r_sa = 0.0
            for prob, s_next, reward, done in transitions:
                r_sa += prob * reward
            r[s, a] = r_sa
    return r


def build_uniform_cost_table(env, bad_states=None, cost_bad=1.0):
    """
    Example cost table: cost 1 when entering any 'bad' state, else 0.
    bad_states: iterable of state indices.
    This is just a simple helper; in your experiments you'll want a
    more meaningful c(s,a).
    """
    import gymnasium as gym
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    nS, nA = env.observation_space.n, env.action_space.n
    c = np.zeros((nS, nA), dtype=np.float32)
    if bad_states is None:
        return c
    bad_states = set(bad_states)
    P = env.unwrapped.P
    for s in range(nS):
        for a in range(nA):
            transitions = P[s][a]
            cost_sa = 0.0
            for prob, s_next, reward, done in transitions:
                if s_next in bad_states:
                    cost_sa += prob * cost_bad
            c[s, a] = cost_sa
    return c