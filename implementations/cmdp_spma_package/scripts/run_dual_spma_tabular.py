
# ==============================================================================
# FILE: scripts/run_dual_spma_tabular.py 
# ==============================================================================
import numpy as np
import gymnasium as gym
from cmdp_spma import (
    PolicyOracleSPMA, 
    OracleConfig, 
    TabularEstimator,
    EntropyRegularizedObjective,
    build_reward_table,
)

def run_dual_spma_tabular_v2(
    make_env,
    objective,
    K_outer=20,
    y_init=None,
    gamma=0.99,
    alpha_y=0.5,
    cfg_kwargs=None,
    K_inner=3,
    rollout_steps=2048,
    seed0=0,
    persistent_policy=True,
    track_original_reward=False,
    reward_table=None,  # For computing J_r under original reward
):
    """
    Generic Dual-SPMA outer loop with enhanced logging.
    
    NEW logging:
        - f_d: primal objective f(d_hat)
        - L: dual objective / saddle value
        - original_J_r: average return under original reward (if tracked)
        - wall_time: cumulative wall-clock time
    """
    base_env = make_env()
    assert isinstance(base_env.observation_space, gym.spaces.Discrete)
    assert isinstance(base_env.action_space, gym.spaces.Discrete)
    nS, nA = base_env.observation_space.n, base_env.action_space.n
    
    if y_init is None:
        y = np.zeros((nS, nA), dtype=np.float32)
    else:
        y = np.array(y_init, dtype=np.float32)
    
    est = TabularEstimator(nS, nA, gamma)
    cfg_dict = dict(discrete=True, gamma=gamma, steps_per_rollout=rollout_steps, 
                    K_inner=K_inner, persistent_policy=persistent_policy)
    if cfg_kwargs:
        cfg_dict.update(cfg_kwargs)
    cfg = OracleConfig(**cfg_dict)
    oracle = PolicyOracleSPMA(make_env, est, cfg)
    
    history = {
        "L": [],           # Saddle value: y·d - f*(y)
        "f_d": [],         # Primal objective: f(d_hat)
        "y_norm": [],
        "sum_d": [],
        "y_dot_d": [],
        "original_J_r": [],  # J_r under original reward
        "wall_time": [],     # Cumulative time
    }
    
    start_time = time.time()
    
    for k in range(K_outer):
        est.reset()  # Per-iteration occupancy estimation
        
        pol_snapshot, d_hat, info = oracle.improve(
            y, K=K_inner, rollout_steps=rollout_steps, seed=seed0 + k,
            track_original_reward=track_original_reward
        )
        
        # Dual gradient: d_hat - grad f*(y)
        grad_f_star_y = objective.grad_f_star(y)
        grad_y = d_hat - grad_f_star_y
        y = y + alpha_y * grad_y
        
        # Compute metrics
        L = float((y * d_hat).sum() - objective.f_star(y))
        f_d = objective.f(d_hat) if hasattr(objective, 'f') else float('nan')
        
        # Original reward (if we have reward table)
        if reward_table is not None:
            original_J_r = float((reward_table * d_hat).sum())
        else:
            original_J_r = float('nan')
        
        history["L"].append(L)
        history["f_d"].append(f_d)
        history["y_norm"].append(float(np.linalg.norm(y)))
        history["sum_d"].append(float(d_hat.sum()))
        history["y_dot_d"].append(float((y * d_hat).sum()))
        history["original_J_r"].append(original_J_r)
        history["wall_time"].append(time.time() - start_time)
        
        print(f"[outer {k:2d}] L={L:.4f}, f(d)={f_d:.4f}, J_r={original_J_r:.4f}, "
              f"||y||={history['y_norm'][-1]:.3f}, sum_d={history['sum_d'][-1]:.4f}, "
              f"time={history['wall_time'][-1]:.1f}s")
    
    return y, history




