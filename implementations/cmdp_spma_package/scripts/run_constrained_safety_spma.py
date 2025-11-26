
# ==============================================================================
# FILE: scripts/run_constrained_safety_spma.py 
# ==============================================================================
# Missing at top:
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

def run_constrained_safety_spma(
    make_env,
    reward_table,
    cost_table,
    tau,
    K_outer=20,
    gamma=0.99,
    alpha_lambda=0.5,
    lambda_init=0.0,
    cfg_kwargs=None,
    K_inner=3,
    rollout_steps=2048,
    seed0=0,
    persistent_policy=True,
):
    """
    Update: Constrained safety with enhanced logging.

    Dual-SPMA for constrained safety:
        max_π J_r(π)  s.t. J_c(π) <= tau
    L(π,λ) = J_r(π) + λ (τ - J_c(π)), λ >= 0.
    
    Policy update uses shaped reward r_λ = r - λ c via y_table = λ c - r.
    Dual update: λ_{k+1} = [ λ_k + α (J_c(π_k) - tau) ]_+.
    
    Args:
        make_env: () -> gym.Env factory.
        reward_table: (nS, nA) array of r(s,a).
        cost_table: (nS, nA) array of c(s,a).
        tau: constraint threshold (max allowed expected cost).
        K_outer: number of outer iterations.
        gamma: discount.
        alpha_lambda: dual step size for λ.
        lambda_init: initial λ value.
        cfg_kwargs: dict to override OracleConfig fields.
        K_inner: SPMA inner iterations per outer step.
        rollout_steps: env steps per inner iteration.
        seed0: base random seed.
    
    Returns:
        lam: final λ
        history: dict of logged values
    """
    import gymnasium as gym
    
    env0 = make_env()
    nS, nA = env0.observation_space.n, env0.action_space.n
    
    objective = ConstrainedSafetyObjective(reward_table, cost_table, tau)
    lam = float(lambda_init)
    
    est = TabularEstimator(nS, nA, gamma)
    cfg_dict = dict(discrete=True, gamma=gamma, steps_per_rollout=rollout_steps,
                    K_inner=K_inner, persistent_policy=persistent_policy)
    if cfg_kwargs:
        cfg_dict.update(cfg_kwargs)
    cfg = OracleConfig(**cfg_dict)
    oracle = PolicyOracleSPMA(make_env, est, cfg)
    
    history = {
        "lambda": [], "J_r": [], "J_c": [], "constraint_violation": [],
        "wall_time": [],
    }
    
    start_time = time.time()
    
    for k in range(K_outer):
        y_table = objective.build_y_table(lam)
        est.reset()
        
        pol_snapshot, d_hat, info = oracle.improve(
            y_table, K=K_inner, rollout_steps=rollout_steps, seed=seed0 + k
        )
        
        evals = objective.eval_objectives(d_hat)
        J_r, J_c, violation = evals["J_r"], evals["J_c"], evals["constraint_violation"]
        
        lam = objective.project_lambda(lam + alpha_lambda * violation)
        
        history["lambda"].append(lam)
        history["J_r"].append(J_r)
        history["J_c"].append(J_c)
        history["constraint_violation"].append(violation)
        history["wall_time"].append(time.time() - start_time)
        
        print(f"[outer {k:2d}] λ={lam:.3f}, J_r={J_r:.4f}, J_c={J_c:.4f}, "
              f"viol={violation:.4f}, time={history['wall_time'][-1]:.1f}s")
    
    return lam, history


