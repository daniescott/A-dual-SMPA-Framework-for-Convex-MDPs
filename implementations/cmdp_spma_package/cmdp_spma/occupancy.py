
import numpy as np

class TabularEstimator:
    """
    Unbiased discounted occupancy estimator:
        d_hat(s,a) = (1-gamma)/N * sum_i sum_t gamma^t * 1{(s_t,a_t)=(s,a)}
    N is the number of episodes (trajectories).
    """
    def __init__(self, nS:int, nA:int, gamma:float):
        self.nS, self.nA, self.gamma = nS, nA, gamma
        self.table = np.zeros((nS, nA), dtype=np.float64)
        self.N = 0  # number of episodes aggregated
   
    def reset(self):
        """Reset the estimator for a new outer iteration."""
        self.table.fill(0.0)
        self.N = 0

    def update_from_batch(self, obs, acts, dones):
        # count episodes in this batch (we assume batch starts with a fresh reset before first step)
        n_eps = int(1 + np.sum(dones > 0.5))
        self.N += max(n_eps, 1)
        gpow = 1.0
        for t in range(len(acts)):
            s = int(obs[t])
            a = int(acts[t])
            self.table[s,a] += (1.0 - self.gamma) * gpow
            gpow *= self.gamma
            if dones[t] > 0.5:
                gpow = 1.0  # reset discount when episode ends
    
    def value(self):
        if self.N == 0:
            return np.zeros_like(self.table)
        return self.table / float(self.N)
    
    def y_dot_d(self, y_table):
        d = self.value()
        return float((y_table * d).sum())
    

class FeatureEstimator:
    """
    Discounted feature expectations:
        E[phi] = (1-gamma)/N * sum_i sum_t gamma^t * phi(s_t,a_t)
    N is the number of episodes (trajectories).
    """
    def __init__(self, phi_fn, d:int, gamma:float):
        self.phi_fn, self.d, self.gamma = phi_fn, d, gamma
        self.sum = np.zeros(d, dtype=np.float64)
        self.N = 0
    def reset(self):
        """Reset the estimator for a new outer iteration.""" // i
        self.sum.fill(0.0)
        self.N = 0

    def update_from_batch(self, obs, acts, dones):
        n_eps = int(1 + np.sum(dones > 0.5))
        self.N += max(n_eps, 1)
        gpow = 1.0
        for t in range(len(acts)):
            phi = self.phi_fn(obs[t], acts[t])
            self.sum += (1.0 - self.gamma) * gpow * phi
            gpow *= self.gamma
            if dones[t] > 0.5:
                gpow = 1.0
    
    def value(self):
        if self.N == 0:
            return np.zeros_like(self.sum)
        return self.sum / float(self.N)
    
    def y_dot_d(self, w):
        ephi = self.value()
        return float(np.dot(ephi, w))
