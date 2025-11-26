import numpy as np
import gymnasium as gym
from cmdp_spma import (
    PolicyOracleSPMA,
    OracleConfig,
    TabularEstimator,
    ConstrainedSafetyObjective,
    build_reward_table,
    build_uniform_cost_table,
)
import run_constrained_safety_spma

def constrained_demo_v2():
    """Constrained safety demo with enhanced logging."""
    def make_env():
        return gym.make("FrozenLake-v1", is_slippery=False)
    
    env = make_env()
    r_table = build_reward_table(env)
    bad_states = [12, 13, 14, 15]
    c_table = build_uniform_cost_table(env, bad_states=bad_states, cost_bad=1.0)
    
    tau = 0.2
    lam_final, hist = run_constrained_safety_spma(
        make_env,
        reward_table=r_table,
        cost_table=c_table,
        tau=tau,
        K_outer=10,
        gamma=0.99,
        alpha_lambda=0.5,
        cfg_kwargs=dict(steps_per_rollout=512, K_inner=2),
        persistent_policy=True,
    )
    return lam_final, hist