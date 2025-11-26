# ==============================================================================
# FILE: cmdp_spma/objectives.py 
# 
# ==============================================================================
import numpy as np

# import numpy as np

class EntropyRegularizedObjective:
    """
    f(d) = -<r,d> + alpha * sum_i d_i log d_i  (occupancy entropy version).
    Conjugate: f*(y) = alpha * log sum_i exp( (y_i + r_i)/alpha ).
    Gradient: grad_f*(y) = softmax( (y + r) / alpha ).
    
    Note: This is the conjugate for occupancy-entropy f(d) = -<r,d> + α Σ d log d,
    which is slightly different from the policy-entropy formulation in Appendix A 
    but still convex and good enough for experiments.
    """
    def __init__(self, reward_table: np.ndarray, alpha: float):
        self.r = np.asarray(reward_table, dtype=np.float32)
        self.alpha = float(alpha)
    
    def f_star(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float32)
        z = (y + self.r) / self.alpha
        # numerically stable log-sum-exp
        z_max = np.max(z)
        return float(self.alpha * (z_max + np.log(np.exp(z - z_max).sum())))
    
    def grad_f_star(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        z = (y + self.r) / self.alpha
        z_max = np.max(z)
        exps = np.exp(z - z_max)
        soft = exps / exps.sum()
        return soft.reshape(y.shape)


class QuadraticObjective:
    """
    Simple quadratic regularizer: f*(y) = (lam/2) ||y||^2
    Gradient: grad_f*(y) = lam * y
    
    This corresponds to f(d) = (1/(2*lam)) ||d||^2 (squared norm penalty).
    """
    def __init__(self, lam: float):
        self.lam = float(lam)
    
    def f_star(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float32)
        return float(0.5 * self.lam * (y**2).sum())
    
    def grad_f_star(self, y: np.ndarray) -> np.ndarray:
        return self.lam * np.asarray(y, dtype=np.float32)


class ConstrainedSafetyObjective:
    """
    Lagrangian CMDP objective for:
        max_π J_r(π)  s.t. J_c(π) <= tau
    
    Lagrangian: L(π, λ) = J_r(π) + λ (τ - J_c(π)),  λ >= 0
    
    We use a scalar dual variable λ, but translate it to a y_table for the 
    policy oracle via: y_table = λ * c - r, so that r_y = -y = r - λc.
    
    The "f*" here is the conjugate of the indicator f(d) = 0 if <c,d> <= τ, else ∞.
    This doesn't have a smooth conjugate, so we work with the Lagrangian directly.
    """
    def __init__(self, reward_table: np.ndarray, cost_table: np.ndarray, tau: float):
        self.r = np.asarray(reward_table, dtype=np.float32)
        self.c = np.asarray(cost_table, dtype=np.float32)
        self.tau = float(tau)
    
    def build_y_table(self, lam: float) -> np.ndarray:
        """Convert scalar λ to y_table for shaped reward r_y = -y = r - λc."""
        return lam * self.c - self.r
    
    def dual_gradient(self, d_hat: np.ndarray) -> float:
        """Gradient w.r.t. λ: J_c(π) - τ = <c, d_hat> - τ."""
        J_c = float((self.c * d_hat).sum())
        return J_c - self.tau
    
    def project_lambda(self, lam: float) -> float:
        """Project λ onto [0, ∞)."""
        return max(0.0, lam)
    
    def eval_objectives(self, d_hat: np.ndarray) -> dict:
        """Evaluate J_r, J_c, and constraint violation."""
        J_r = float((self.r * d_hat).sum())
        J_c = float((self.c * d_hat).sum())
        return {
            "J_r": J_r,
            "J_c": J_c,
            "constraint_violation": J_c - self.tau
        }