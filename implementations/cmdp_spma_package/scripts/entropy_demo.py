from cmdp_spma import (
    PolicyOracleSPMA,
    OracleConfig,
    TabularEstimator,
    ConstrainedSafetyObjective,
    build_reward_table,
    build_uniform_cost_table,
    EntropyRegularizedObjective
)
import run_dual_spma_tabular

def entropy_demo_v2():
    """Demo with enhanced logging."""
    def make_env():
        return gym.make("FrozenLake-v1", is_slippery=False)
    
    env = make_env()
    r_table = build_reward_table(env)
    alpha = 0.1
    obj = EntropyRegularizedObjective(r_table, alpha=alpha)
    
    y_final, hist = run_dual_spma_tabular(
        make_env, obj,
        K_outer=10,
        gamma=0.99,
        alpha_y=0.5,
        cfg_kwargs=dict(steps_per_rollout=512, K_inner=2),
        persistent_policy=True,
        reward_table=r_table,
    )
    return y_final, hist