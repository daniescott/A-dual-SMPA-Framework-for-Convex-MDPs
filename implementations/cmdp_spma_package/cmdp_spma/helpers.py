# ==============================================================================
# FILE: cmdp_spma/helpers.py 
# build_reward_table, build_uniform_cost_table
# ==============================================================================
import numpy as np
import torch

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

def estimate_Jr_Jc(make_env, actor, reward_table, cost_table,
                   gamma=0.99, n_episodes=32, max_steps=200, device="cpu"):
    """
    Monte Carlo estimates of Jr and Jc under a fixed actor π.
    Uses the tabular reward_table r(s,a) and cost_table c(s,a).
    
    Args:
        make_env: environment factory function
        actor: policy network (ActorDiscrete)
        reward_table: (nS, nA) array of r(s,a)
        cost_table: (nS, nA) array of c(s,a)
        gamma: discount factor
        n_episodes: number of episodes for Monte Carlo estimation
        max_steps: maximum steps per episode
        device: torch device
    
    Returns:
        Jr: estimated discounted reward return
        Jc: estimated discounted cost return
    """
    env = make_env()
    Jr = 0.0
    Jc = 0.0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        gpow = 1.0
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a, _ = actor.act(obs_t)
            a_np = int(a.squeeze(0).cpu().item())
            s = int(obs)
            
            # Accumulate discounted rewards and costs from tables
            Jr += gpow * float(reward_table[s, a_np])
            Jc += gpow * float(cost_table[s, a_np])
            
            obs, _, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            gpow *= gamma
            
            if done:
                break
    
    Jr /= float(n_episodes)
    Jc /= float(n_episodes)
    return Jr, Jc