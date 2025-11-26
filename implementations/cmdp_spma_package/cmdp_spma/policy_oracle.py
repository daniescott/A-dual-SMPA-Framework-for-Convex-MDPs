
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import gymnasium as gym

from .nets import ActorDiscrete, ActorGaussian, Critic
from .rollout import collect_rollouts
from .spma_losses import spma_actor_loss, mse_loss
from .line_search import _flatten_grads, _flatten_params, _set_params_from_vector, armijo_backtracking
from .shaped_reward_env import TabularShapedReward, FeatureShapedReward
from .occupancy import TabularEstimator, FeatureEstimator

@dataclass
class OracleConfig:
    device: str = "cpu"
    gamma: float = 0.99
    lam: float = 0.95
    inv_eta: float = 5.0            # 1/η in the SPMA term
    max_grad_norm: float = 1.0
    armijo_c: float = 1e-4
    armijo_beta: float = 0.5
    armijo_init: float = 1.0
    armijo_max_steps: int = 15
    critic_lr: float = 3e-4
    K_inner: int = 5
    steps_per_rollout: int = 2048
    discrete: bool = True
    hidden: Tuple[int,int] = (64,64)
    persistent_policy: bool = True  #if True, keep actor/critic across iterations

class PolicyOracleSPMA:
    """
    Policy oracle with OPTIONAL policy persistence.
    
    If cfg.persistent_policy=True:
        - Actor and critic are created once in __init__
        - improve(y) updates the SAME policy with new shaped reward
        - Shows how policy gradually adapts as dual y changes
    
    If cfg.persistent_policy=False:
        - Actor and critic are recreated each improve() call (v1 behavior)
    """
    def __init__(self, 
                 env_maker: Callable[[], gym.Env],
                 estimator: Union[TabularEstimator, FeatureEstimator],
                 cfg: OracleConfig):
        self.env_maker = env_maker
        self.estimator = estimator
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Build a sample env to get dimensions
        sample_env = env_maker()
        self._obs_dim, self._act_dim, self._n_actions = self._get_dims(sample_env)
        
        # Persistent actor/critic (if enabled)
        self.actor = None
        self.critic = None
        if cfg.persistent_policy:
            self.actor, self.critic = self._build_models()
        
        # Shaped reward wrapper (reused across iterations)
        self._shaped_env = None
        self._original_env = None
    
    def _get_dims(self, env):
        obs_space = env.observation_space
        act_space = env.action_space
        if hasattr(act_space, "n"):
            n_actions = act_space.n
            act_dim = None
            if hasattr(obs_space, "shape") and obs_space.shape is not None:
                obs_dim = int(np.prod(obs_space.shape))
            else:
                obs_dim = obs_space.n
        else:
            n_actions = None
            act_dim = int(np.prod(act_space.shape))
            obs_dim = int(np.prod(obs_space.shape))
        return obs_dim, act_dim, n_actions
    
    def _build_models(self):
        if self._n_actions is not None:
            actor = ActorDiscrete(self._obs_dim, self._n_actions, hidden=self.cfg.hidden).to(self.device)
        else:
            actor = ActorGaussian(self._obs_dim, self._act_dim, hidden=self.cfg.hidden).to(self.device)
        critic = Critic(self._obs_dim, hidden=self.cfg.hidden).to(self.device)
        return actor, critic
    
    def _make_or_update_shaped_env(self, y):
        """Create or update the shaped reward environment."""
        if self._n_actions is not None:
            # Tabular case
            if self._shaped_env is None:
                base = self.env_maker()
                self._shaped_env = TabularShapedReward(base, y, self.cfg.gamma)
                self._original_env = self.env_maker()  # Keep original for logging
            else:
                self._shaped_env.set_y(y)
        else:
            # Feature case
            phi_fn, w = y
            if self._shaped_env is None:
                base = self.env_maker()
                self._shaped_env = FeatureShapedReward(base, phi_fn, w, self.cfg.gamma)
                self._original_env = self.env_maker()
            else:
                self._shaped_env.set_w(w)
        return self._shaped_env
    
    def _update_actor_spma(self, actor, batch):
        obs = batch["obs"]
        acts = batch["acts"]
        adv = batch["adv"]
        old_logp = batch["old_logp"]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(actor, ActorDiscrete):
            acts_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        else:
            acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device).detach()

        def loss_fn():
            logp_new = actor.log_prob(obs_t, acts_t)
            return spma_actor_loss(logp_new, old_logp_t, adv_t, self.cfg.inv_eta)

        L0 = float(loss_fn().detach().cpu())
        for p in actor.parameters():
            if p.grad is not None:
                p.grad.zero_()
        L = loss_fn()
        L.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        g = _flatten_grads(actor)
        d = -g
        gTd = float((g * d).sum().detach().cpu())
        if gTd >= 0:
            gTd = -1.0

        accepted, alpha, Lnew = armijo_backtracking(
            loss_fn, actor, L0, gTd, d,
            alpha0=self.cfg.armijo_init, beta=self.cfg.armijo_beta, c=self.cfg.armijo_c,
            max_steps=self.cfg.armijo_max_steps
        )
        if not accepted:
            step = 1e-3
            with torch.no_grad():
                vec = _flatten_params(actor)
                vec = vec + step * d
                _set_params_from_vector(actor, vec)
            Lnew = float(loss_fn().detach().cpu())
        return L0, Lnew

    def _update_critic(self, critic, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(critic.parameters(), lr=self.cfg.critic_lr)
        for _ in range(10):
            v = critic.value(obs)
            loss = mse_loss(v, ret)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach().cpu())

    def improve(self, y, K: Optional[int]=None, rollout_steps: Optional[int]=None, 
                seed: Optional[int]=None, track_original_reward: bool = False):
        """
        Perform K SPMA updates under shaped reward r_y = -y.
        
        Args:
            y: dual variable (table for tabular, (phi_fn, w) for feature)
            K: number of inner iterations
            rollout_steps: steps per rollout
            seed: random seed
            track_original_reward: if True, also record reward under original env
        
        Returns:
            policy_snapshot, d_hat, logs
        """
        if K is None: K = self.cfg.K_inner
        if rollout_steps is None: rollout_steps = self.cfg.steps_per_rollout
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Get or create actor/critic
        if self.cfg.persistent_policy:
            actor, critic = self.actor, self.critic
        else:
            actor, critic = self._build_models()
        
        # Update shaped reward env
        env = self._make_or_update_shaped_env(y)
        
        logs = {
            "actor_loss_before": [], "actor_loss_after": [], "critic_loss": [],
            "original_return_per_step": []  # track original reward
        }
        
        for _ in range(K):
            if track_original_reward and self._original_env is not None:
                batch = collect_rollouts_dual_reward(
                    env, self._original_env, actor, critic, rollout_steps,
                    device=self.device, gamma=self.cfg.gamma, lam=self.cfg.lam
                )
                # Compute average return under original reward
                orig_return = float(batch["rews_original"].sum()) / max(1, batch["dones"].sum())
                logs["original_return_per_step"].append(float(batch["rews_original"].mean()))
            else:
                batch = collect_rollouts(env, actor, critic, rollout_steps,
                                        device=self.device, gamma=self.cfg.gamma, lam=self.cfg.lam)
            
            L0, Lnew = self._update_actor_spma(actor, batch)
            closs = self._update_critic(critic, batch)
            logs["actor_loss_before"].append(L0)
            logs["actor_loss_after"].append(Lnew)
            logs["critic_loss"].append(closs)
            self.estimator.update_from_batch(batch["obs"], batch["acts"], batch["dones"])
        
        # Get occupancy estimate
        if isinstance(self.estimator, TabularEstimator):
            d_hat = self.estimator.value()
            ydotd = self.estimator.y_dot_d(y)
        else:
            ephi = self.estimator.value()
            if isinstance(y, tuple):
                _, w = y
                ydotd = self.estimator.y_dot_d(w)
            else:
                ydotd = float(np.nan)
            d_hat = ephi
        
        policy_snapshot = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        return policy_snapshot, d_hat, {"y_dot_d": ydotd, **logs}


