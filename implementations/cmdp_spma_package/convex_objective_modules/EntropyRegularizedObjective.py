import numpy as np
from scipy.special import logsumexp  # or implement by hand

# We represent y as an (nS, nA) array.
class EntropyRegularizedObjective:
    """
    f(d) = -<r,d> + alpha * sum_i d_i log d_i (occupancy entropy version).
    Conjugate: f*(y) = alpha * log sum_i exp( (y_i + r_i)/alpha ).
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

# Note: This is the conjugate for occupancy‑entropy 
# f(d)=−⟨r,d⟩+α∑dlog⁡d
# f(d)=−⟨r,d⟩+α∑dlogd, which is slightly different from the policy‑entropy formulation in Appendix A but still convex and good enough for experiment
