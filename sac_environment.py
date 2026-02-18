import numpy as np
import gym
from gym import spaces


class NetworkOptimizationSACEnv(gym.Env):
    """
    Gym environment for network optimization with:
    - Normalized, scale-invariant state representation
    - Bounded reward function
    - Stochastic action rounding
    - Bandwidth estimation tracking
    """
    
    def __init__(self, simulator):
        super().__init__()
        
        self.simulator = simulator
        
        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # State space: [u_t, c_t, Δu_t, Δc_t]
        # u_t and c_t are [0, 1], deltas are [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 10
        self.current_step = 0
        
        # Thread limits
        self.cc_min = 1
        self.cc_max = 32
        
        # Bandwidth estimation (updated during episode)
        self.BW_est = 1000.0  # Initial estimate
        self.TPT_est = 50.0   # Initial estimate
        
        # State tracking
        self.current_throughput = 0.0
        self.current_concurrency = 1
        self.packet_loss_rate = 0.0
        
        # Previous state (for deltas)
        self.prev_u_t = 0.0
        self.prev_c_t = 0.0
        
        # Reward function hyperparameters (tunable)
        self.lambda_cc = 0.15    # Concurrency penalty weight
        self.beta_cc = 2.0       # Concurrency penalty exponent (quadratic)
        self.mu_loss = 5.0       # Packet loss penalty weight
        self.nu_stability = 0.05 # Action smoothness penalty
        
    def reset(self, preserve_estimates=False, **kwargs):
        """Reset environment and estimate bandwidth
        
        Args:
            preserve_estimates: If True, keep current BW_est/TPT_est values
            **kwargs: Additional arguments (seed, options, etc.) - accepted for compatibility but not used
        """
        # Reset simulator (will randomize if enabled)
        sim_info = self.simulator.reset()
        
        # Only update estimates if we haven't done exploration yet
        # This prevents overwriting exploration results
        if not preserve_estimates or self.BW_est == 1000.0:
            self.BW_est = sim_info['bw_est']
            self.TPT_est = sim_info['tpt_est']
        
        # Random initial concurrency
        self.current_concurrency = np.random.randint(3, self.cc_max - 1)
        
        # Get initial throughput
        throughput, loss = self.simulator.get_utility_value([self.current_concurrency])
        self.current_throughput = throughput
        self.packet_loss_rate = loss
        
        # Initialize state
        self.current_step = 0
        state = self._get_normalized_state()
        
        # Store for next delta
        self.prev_u_t = state[0]
        self.prev_c_t = state[1]
        
        return state
    
    def step(self, action):
        """
        Execute action and return normalized state, bounded reward
        
        Args:
            action: continuous value in [-1, 1]
        """
        delta_cc = int(action[0] * 5)  # Scale to [-5, +5]
        cc_actual = np.clip(self.current_concurrency + delta_cc, self.cc_min, self.cc_max)
        
        # Store previous concurrency for stability penalty
        prev_cc = self.current_concurrency
        self.current_concurrency = cc_actual
        
        # Execute in simulator
        throughput, packet_loss = self.simulator.get_utility_value([cc_actual])
        self.current_throughput = throughput
        self.packet_loss_rate = packet_loss
        
        # Update bandwidth estimate (exponential moving average)
        # This can both increase and decrease, handling sudden drops
        self.BW_est = 0.9 * self.BW_est + 0.1 * throughput
        self.BW_est = max(self.BW_est, 1.0)  # Floor to avoid division by zero
        
        # Get normalized state
        state = self._get_normalized_state()
        
        # Compute bounded reward
        reward = self._compute_reward(state, prev_cc)
        
        # Update previous state
        self.prev_u_t = state[0]
        self.prev_c_t = state[1]
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': self.current_throughput,
            'concurrency': self.current_concurrency,
            'packet_loss': self.packet_loss_rate,
            'bw_est': self.BW_est
        }
        
        return state, reward, done, info
    
    def _get_normalized_state(self):
        """
        Compute scale-invariant state: [u_t, c_t, Δu_t, Δc_t]
        
        Returns:
            state: numpy array of shape (4,)
        """
        # Normalized throughput utilization
        u_t = min(1.0, self.current_throughput / (self.BW_est + 1e-8))
        
        # Normalized concurrency
        c_t = self.current_concurrency / self.cc_max
        
        # Deltas (change from previous step)
        delta_u = u_t - self.prev_u_t
        delta_c = c_t - self.prev_c_t
        
        # Clip deltas to [-1, 1] for safety
        delta_u = np.clip(delta_u, -1.0, 1.0)
        delta_c = np.clip(delta_c, -1.0, 1.0)
        
        return np.array([u_t, c_t, delta_u, delta_c], dtype=np.float32)
    
    def _compute_reward(self, state, prev_cc):
        """
        Compute bounded, normalized reward
        
        Reward = utilization_bonus - concurrency_penalty - loss_penalty - stability_penalty
        
        Args:
            state: current normalized state [u_t, c_t, Δu_t, Δc_t]
            prev_cc: previous concurrency (for stability)
        """
        u_t, c_t, delta_u, delta_c = state
        
        # 1. Utilization bonus (logarithmic for diminishing returns)
        # log(1 + β*u) encourages high utilization but with diminishing returns
        utilization_bonus = np.log1p(5.0 * u_t)  # log(1 + 5*u)
        
        # 2. Concurrency penalty (quadratic: c^2)
        # Penalizes high concurrency to reduce CPU overhead
        concurrency_penalty = self.lambda_cc * (c_t ** self.beta_cc)
        
        # 3. Packet loss penalty (linear)
        # Heavily penalize packet loss for fairness
        loss_penalty = self.mu_loss * self.packet_loss_rate
        
        # 4. Stability penalty (discourage rapid changes)
        # |Δc| penalizes oscillation
        stability_penalty = self.nu_stability * abs(delta_c)
        
        # Combine
        reward = utilization_bonus - concurrency_penalty - loss_penalty - stability_penalty
        
        # Clip to bounded range for SAC stability
        reward = np.clip(reward, -5.0, 5.0)
        
        return reward
    
    def exploration_phase(self, duration=100):
        """
        Run random exploration to estimate BW_est and TPT_est
        This should be called before training on a new network
        
        Args:
            duration: number of random samples
        """
        throughputs = []
        concurrencies = []
        
        for _ in range(duration):
            cc = np.random.randint(self.cc_min, self.cc_max + 1)
            throughput, _ = self.simulator.get_utility_value([cc])
            
            throughputs.append(throughput)
            concurrencies.append(cc)
        
        # Estimate bandwidth (max observed)
        self.BW_est = max(throughputs)
        
        # Estimate per-thread throughput
        tpt_estimates = [t / c for t, c in zip(throughputs, concurrencies)]
        self.TPT_est = max(tpt_estimates)
        
        return {
            'BW_est': self.BW_est,
            'TPT_est': self.TPT_est
        }


class ExplorationWrapper(gym.Env):
    """
    Wrapper that adds bandwidth estimation before each episode
    This ensures the agent starts with good normalization parameters
    """
    
    def __init__(self, env, exploration_steps=20):
        self.env = env
        self.exploration_steps = exploration_steps
        
        # Copy spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def reset(self, **kwargs):
        """Reset with exploration phase"""
        # Do light exploration (faster than full estimation)
        if self.exploration_steps > 0:
            self.env.exploration_phase(self.exploration_steps)
        
        # Reset but preserve the estimates we just computed
        # Pass through any additional kwargs (seed, options, etc.)
        return self.env.reset(preserve_estimates=True, **kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def __getattr__(self, name):
        """Delegate other methods to wrapped env"""
        return getattr(self.env, name)
