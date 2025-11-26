
import numpy as np
import gymnasium as gym

from cmdp_spma import (
    TabularEstimator,
    FeatureEstimator,
    TabularShapedReward,
    PolicyOracleSPMA,
    OracleConfig,
)

def test_simple_sum_to_one():
    """Test that TabularEstimator produces occupancy measures that sum to ~1."""
    gamma = 0.9
    est = TabularEstimator(2, 2, gamma)
    obs = np.array([0,0,1,1], dtype=np.int64)
    acts = np.array([0,1,0,1], dtype=np.int64)
    dones = np.array([0,0,0,1], dtype=np.float32)
    est.update_from_batch(obs, acts, dones)
    d = est.value()
    s = d.sum()
    assert 0.9 <= s <= 1.1, f"sum d should be ~1, got {s}"
    print("test_simple_sum_to_one: PASSED")


def test_feature_estimator_indicator():
    """
    Test that for φ(s,a) = e_i (indicator), the estimator recovers the right coordinate.
    
    If φ(s,a) = e_{s*nA + a} (one-hot over (s,a) pairs), then E[φ]_i = d(s,a)
    for the corresponding (s,a) pair.
    """
    nS, nA = 3, 2
    gamma = 0.9
    
    # One-hot feature function
    def phi_onehot(s, a):
        vec = np.zeros(nS * nA, dtype=np.float32)
        idx = int(s) * nA + int(a)
        vec[idx] = 1.0
        return vec
    
    est = FeatureEstimator(phi_onehot, nS * nA, gamma)
    
    # Simulate a trajectory: (0,0), (1,1), (2,0), done
    obs = np.array([0, 1, 2], dtype=np.int64)
    acts = np.array([0, 1, 0], dtype=np.int64)
    dones = np.array([0, 0, 1], dtype=np.float32)
    
    est.update_from_batch(obs, acts, dones)
    ephi = est.value()
    
    # The feature expectation should have non-zero entries only at visited (s,a) pairs
    # and should match the discounted occupancy
    expected_indices = [0*nA+0, 1*nA+1, 2*nA+0]  # indices 0, 3, 4
    
    for i in range(nS * nA):
        if i in expected_indices:
            assert ephi[i] > 0, f"E[φ][{i}] should be > 0 for visited (s,a)"
        else:
            assert ephi[i] == 0, f"E[φ][{i}] should be 0 for unvisited (s,a)"
    
    # Sum should be ~1 (it's an occupancy measure)
    assert 0.9 <= ephi.sum() <= 1.1, f"sum E[φ] should be ~1, got {ephi.sum()}"
    
    print("test_feature_estimator_indicator: PASSED")


def test_shaped_reward_affects_returns():
    """
    Test that replacing reward with -y actually affects returns as expected.
    
    In a simple 2-state chain MDP:
        State 0 --a=0--> State 1 (done)
    
    If y[0,0] = 5, then shaped reward = -5, so return should be -5.
    If y[0,0] = -3, then shaped reward = 3, so return should be 3.
    """
    # Create a minimal chain MDP
    class TinyChainMDP(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(2)
            self.action_space = gym.spaces.Discrete(1)
            self.state = 0
        
        def reset(self, **kwargs):
            self.state = 0
            return self.state, {}
        
        def step(self, a):
            # From state 0, go to state 1 and terminate
            if self.state == 0:
                self.state = 1
                return self.state, 1.0, True, False, {}  # Original reward = 1
            return self.state, 0.0, True, False, {}
    
    env = TinyChainMDP()
    gamma = 0.99
    
    # Test 1: y[0,0] = 5 => shaped reward = -5
    y1 = np.array([[5.0], [0.0]], dtype=np.float32)
    shaped_env1 = TabularShapedReward(TinyChainMDP(), y1, gamma)
    shaped_env1.reset()
    _, r1, _, _, _ = shaped_env1.step(0)
    assert r1 == -5.0, f"Expected shaped reward -5, got {r1}"
    
    # Test 2: y[0,0] = -3 => shaped reward = 3
    y2 = np.array([[-3.0], [0.0]], dtype=np.float32)
    shaped_env2 = TabularShapedReward(TinyChainMDP(), y2, gamma)
    shaped_env2.reset()
    _, r2, _, _, _ = shaped_env2.step(0)
    assert r2 == 3.0, f"Expected shaped reward 3, got {r2}"
    
    print("test_shaped_reward_affects_returns: PASSED")


def test_spma_bandit_convergence():
    """
    SPMA sanity test: in a toy bandit, check that the policy converges to the arm
    with the largest reward.
    
    3-arm bandit:
        - Arm 0: reward 0.1
        - Arm 1: reward 0.9  <-- best
        - Arm 2: reward 0.5
    
    After training with y=0 (no shaping), policy should prefer arm 1.
    """
    class SimpleBandit(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(1)  # Single state
            self.action_space = gym.spaces.Discrete(3)
            self.rewards = [0.1, 0.9, 0.5]
        
        def reset(self, **kwargs):
            return 0, {}
        
        def step(self, a):
            r = self.rewards[int(a)]
            return 0, r, True, False, {}  # Always terminates
    
    def make_bandit():
        return SimpleBandit()
    
    gamma = 0.99
    nS, nA = 1, 3
    y = np.zeros((nS, nA), dtype=np.float32)  # No shaping
    
    est = TabularEstimator(nS, nA, gamma)
    cfg = OracleConfig(discrete=True, steps_per_rollout=500, K_inner=10, gamma=gamma,
                       persistent_policy=True, inv_eta=1.0)  # Lower inv_eta for faster learning
    oracle = PolicyOracleSPMA(make_bandit, est, cfg)
    
    # Train for several iterations
    for _ in range(5):
        est.reset()
        _, d_hat, _ = oracle.improve(y, K=10, rollout_steps=500)
    
    # Check that arm 1 (best) has highest occupancy
    # d_hat[0, a] is the occupancy of action a in state 0
    best_arm = np.argmax(d_hat[0])
    
    # Allow some tolerance - policy should at least strongly prefer arm 1
    print(f"  Occupancy: arm0={d_hat[0,0]:.3f}, arm1={d_hat[0,1]:.3f}, arm2={d_hat[0,2]:.3f}")
    assert d_hat[0, 1] > d_hat[0, 0], "Arm 1 should be preferred over arm 0"
    assert d_hat[0, 1] > d_hat[0, 2], "Arm 1 should be preferred over arm 2"
    
    print("test_spma_bandit_convergence: PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running all tests...")
    print("=" * 60)
    
    test_simple_sum_to_one()
    test_feature_estimator_indicator()
    test_shaped_reward_affects_returns()
    test_spma_bandit_convergence()
    
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)