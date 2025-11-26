# ==============================================================================
# SCRIPT: scripts/run_constrained_safety_npg_pd.py
# ==============================================================================
"""
Constrained-safety CMDP solved by a sample-based NPG-PD baseline.

- Primal: natural policy gradient ascent on r_lambda(s,a) = r(s,a) - lambda*c(s,a)
- Dual: projected subgradient ascent on lambda with step size beta.

This mirrors run_constrained_safety_spma.py but uses NPG instead of SPMA.
"""

import time
import numpy as np
import torch
import gymnasium as gym

# Import from cmdp_spma package
from cmdp_spma import (
    ActorDiscrete,
    Critic,
    TabularShapedReward,
    ConstrainedSafetyObjective,
    build_reward_table,
    build_uniform_cost_table,
)
from cmdp_spma.rollout import collect_rollouts
from cmdp_spma.spma_losses import mse_loss
from cmdp_spma.npg_pd import npg_actor_step_diag
from cmdp_spma.helpers import estimate_Jr_Jc


def run_constrained_safety_npg_pd(
    make_env=None,
    reward_table=None,
    cost_table=None,
    tau=0.1,
    K_outer=30,
    steps_per_rollout=2048,
    gamma=0.99,
    gae_lam=0.95,
    npg_step=0.05,
    fisher_eps=1e-3,
    beta_lambda=0.5,
    lambda_init=0.0,
    n_eval_episodes=32,
    max_eval_steps=200,
    device="cpu",
    seed=0,
):
    """
    Constrained-safety CMDP solved by a sample-based NPG-PD baseline.
    
    Args:
        make_env: environment factory function
        reward_table: (nS, nA) array of r(s,a)
        cost_table: (nS, nA) array of c(s,a)
        tau: safety threshold (max allowed expected cost)
        K_outer: number of outer iterations
        steps_per_rollout: environment steps per iteration
        gamma: discount factor
        gae_lam: GAE lambda
        npg_step: natural policy gradient step size
        fisher_eps: Fisher matrix regularization
        beta_lambda: dual step size for lambda
        lambda_init: initial lambda value
        n_eval_episodes: episodes for evaluating Jr and Jc
        max_eval_steps: max steps per evaluation episode
        device: torch device string
        seed: random seed
    
    Returns:
        actor: trained policy
        history: dict of logged metrics
    """
    # Default environment: FrozenLake
    if make_env is None:
        def make_env():
            return gym.make("FrozenLake-v1", is_slippery=False)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get environment dimensions
    base_env = make_env()
    nS, nA = base_env.observation_space.n, base_env.action_space.n
    
    # Build reward/cost tables if not provided
    if reward_table is None:
        reward_table = build_reward_table(base_env)
    if cost_table is None:
        # Default: last row of FrozenLake as unsafe
        bad_states = [12, 13, 14, 15]  # Adjust for your env
        cost_table = build_uniform_cost_table(base_env, bad_states=bad_states, cost_bad=1.0)
    
    # Create objective (for building y_table and projecting lambda)
    objective = ConstrainedSafetyObjective(reward_table, cost_table, tau)
    
    # Initialize dual variable
    lam = float(lambda_init)
    
    # Initialize actor and critic (PERSISTENT across iterations)
    device = torch.device(device)
    obs_dim = nS  # FrozenLake observations are integers
    hidden = (64, 64)
    
    actor = ActorDiscrete(obs_dim, nA, hidden=hidden).to(device)
    critic = Critic(obs_dim, hidden=hidden).to(device)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    
    # History for logging
    history = {
        "lambda": [],
        "Jr": [],
        "Jc": [],
        "constraint_violation": [],
        "actor_loss_before": [],
        "actor_loss_after": [],
        "wall_time": [],
    }
    
    start_time = time.time()
    
    for k in range(K_outer):
        # --- Build shaped reward environment: r_λ(s,a) = r(s,a) - λ*c(s,a) ---
        y_table = objective.build_y_table(lam)  # y = λc - r, so r_y = -y = r - λc
        env = TabularShapedReward(make_env(), y_table, gamma)
        
        # --- Collect rollouts under shaped reward ---
        batch = collect_rollouts(env, actor, critic, 
                                 steps_per_rollout, device=device,
                                 gamma=gamma, lam=gae_lam)
        
        # --- Critic update (same as in PolicyOracleSPMA) ---
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
        for _ in range(10):
            v = critic.value(obs_t)
            loss_v = mse_loss(v, ret_t)
            critic_opt.zero_grad()
            loss_v.backward()
            critic_opt.step()
        
        # --- Natural PG actor update ---
        L_before, L_after = npg_actor_step_diag(
            actor, batch, step_size=npg_step, fisher_eps=fisher_eps, device=device
        )
        
        # --- Evaluate Jr and Jc under CURRENT policy (on original env) ---
        Jr, Jc = estimate_Jr_Jc(make_env, actor, reward_table, cost_table,
                                gamma=gamma, n_episodes=n_eval_episodes,
                                max_steps=max_eval_steps, device=device)
        
        # --- Dual update: λ_{k+1} = [λ_k + β(J_c - τ)]_+ ---
        violation = Jc - tau
        lam = objective.project_lambda(lam + beta_lambda * violation)
        
        # --- Log metrics ---
        history["lambda"].append(lam)
        history["Jr"].append(Jr)
        history["Jc"].append(Jc)
        history["constraint_violation"].append(violation)
        history["actor_loss_before"].append(L_before)
        history["actor_loss_after"].append(L_after)
        history["wall_time"].append(time.time() - start_time)
        
        print(f"[NPG-PD iter {k:2d}] "
              f"λ={lam:.3f}, Jr={Jr:.4f}, Jc={Jc:.4f}, Jc-τ={violation:.4f}")
    
    return actor, history


def npg_pd_demo():
    """Demo run of NPG-PD on FrozenLake with safety constraints."""
    import gymnasium as gym
    
    def make_env():
        return gym.make("FrozenLake-v1", is_slippery=False)
    
    base_env = make_env()
    r_table = build_reward_table(base_env)
    bad_states = [5, 7, 11, 12]  # Holes in FrozenLake 4x4
    c_table = build_uniform_cost_table(base_env, bad_states=bad_states, cost_bad=1.0)
    
    tau = 0.1  # Safety threshold
    
    actor, history = run_constrained_safety_npg_pd(
        make_env=make_env,
        reward_table=r_table,
        cost_table=c_table,
        tau=tau,
        K_outer=20,
        steps_per_rollout=1024,
        gamma=0.99,
        npg_step=0.05,
        beta_lambda=0.5,
        seed=42,
    )
    
    return actor, history


if __name__ == "__main__":
    print("=" * 60)
    print("Running NPG-PD Constrained Safety Demo")
    print("=" * 60)
    actor, history = npg_pd_demo()
    print("\nDone!")
    print(f"Final λ: {history['lambda'][-1]:.4f}")
    print(f"Final Jr: {history['Jr'][-1]:.4f}")
    print(f"Final Jc: {history['Jc'][-1]:.4f}")