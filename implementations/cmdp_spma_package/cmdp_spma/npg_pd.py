# ==============================================================================
# FILE: cmdp_spma/npg_pd.py
# NPG-PD: Natural Policy Gradient with Primal-Dual updates
# ==============================================================================
"""
NPG-PD baseline implementation for constrained MDPs.

This implements the Natural Policy Gradient - Primal Dual algorithm:
- Primal step: natural policy gradient ascent on J_r(π) - λ J_c(π)
- Dual step: projected subgradient ascent on λ

Reference: Ding et al., "Natural Policy Gradient Primal-Dual Method for Constrained Markov Decision Processes"
"""

import torch
import numpy as np

from .line_search import _flatten_grads, _flatten_params, _set_params_from_vector


def npg_actor_step_diag(actor, batch, step_size=0.05, fisher_eps=1e-3, device="cpu"):
    """
    Approximate natural policy gradient step using a diagonal Fisher matrix.
    
    Given a batch with fields:
        obs, acts, adv
    we compute:
        g = E[ ∇θ log π(a|s) * A ]
        F_diag = E[ (∇θ log π(a|s))^2 ]
    and update θ <- θ + step_size * F_diag^{-1} g.
    
    Args:
        actor: policy network (ActorDiscrete or ActorGaussian)
        batch: dict with 'obs', 'acts', 'adv' keys
        step_size: learning rate for natural gradient step
        fisher_eps: small constant for numerical stability in F^{-1}
        device: torch device
    
    Returns:
        loss_pg_old: policy gradient loss before update
        loss_pg_new: policy gradient loss after update
    """
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    acts = torch.as_tensor(batch["acts"], dtype=torch.long, device=device)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    
    # 1) Vanilla policy gradient
    actor.zero_grad()
    logp = actor.log_prob(obs, acts)
    loss_pg = -(adv.detach() * logp).mean()
    loss_pg.backward()
    g = _flatten_grads(actor)
    
    # 2) Diagonal Fisher: average squared gradients of log π
    B = obs.shape[0]
    F_diag = torch.zeros_like(g)
    for i in range(B):
        actor.zero_grad()
        logp_i = actor.log_prob(obs[i:i+1], acts[i:i+1]).mean()
        logp_i.backward()
        grad_i = _flatten_grads(actor)
        F_diag += grad_i.pow(2)
    F_diag /= float(B)
    
    # 3) Natural gradient step: θ <- θ + step_size * F^{-1} g
    # Note: g is gradient of LOSS (negative reward), so we ADD to maximize reward
    # But loss_pg = -E[adv * logp], so -g points in direction of increasing reward
    nat_grad = g / (F_diag + fisher_eps)
    old_params = _flatten_params(actor)
    new_params = old_params - step_size * nat_grad  # Gradient descent on loss = ascent on reward
    _set_params_from_vector(actor, new_params)
    
    # Compute new loss for logging
    with torch.no_grad():
        logp_new = actor.log_prob(obs, acts)
        loss_pg_new = float((-(adv * logp_new).mean()).cpu().item())
        loss_pg_old = float(loss_pg.detach().cpu().item())
    
    return loss_pg_old, loss_pg_new


def npg_actor_step_diag_lagrangian(actor, batch, lam, reward_table, cost_table,
                                    step_size=0.05, fisher_eps=1e-3, device="cpu"):
    """
    NPG step on the Lagrangian objective: J_r(π) - λ J_c(π).
    
    The advantage is computed as: A_λ(s,a) = A_r(s,a) - λ A_c(s,a)
    where we approximate this by using shaped reward r_λ = r - λc.
    
    This version directly uses the batch advantages which should already
    be computed under the shaped reward r_λ = r - λc.
    
    Args:
        actor: policy network
        batch: dict with 'obs', 'acts', 'adv' (advantages under r_λ)
        lam: current Lagrange multiplier
        reward_table: r(s,a) table (for reference, not used here)
        cost_table: c(s,a) table (for reference, not used here)
        step_size: NPG step size
        fisher_eps: Fisher regularization
        device: torch device
    
    Returns:
        loss_before, loss_after
    """
    # The batch already has advantages computed under shaped reward r_λ = r - λc
    # So we just do a standard NPG step
    return npg_actor_step_diag(actor, batch, step_size=step_size, 
                               fisher_eps=fisher_eps, device=device)