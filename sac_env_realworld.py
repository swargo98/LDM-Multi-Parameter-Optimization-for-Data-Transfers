"""
Real-World SAC Environment
==========================
Drop-in replacement for sac_environment.py when using a real network.

Black-box contract (identical to the simulator):
    throughput, packet_loss_rate = black_box_function([thread_count: int])

This is the ONLY change from the simulator environment:
    - No simulator object
    - No simulator.reset() (network stays live; only episode counter resets)
    - Everything else (normalization, reward, exploration) is identical
"""

import numpy as np
import gym
from gym import spaces


class RealWorldSACEnv(gym.Env):
    """
    SAC environment backed by a real network transfer.

    Replace GeneralizedNetworkSimulator with your black_box_function:

        throughput, packet_loss_rate = black_box_function([thread_count])

    Everything else (state normalization, reward, exploration) is identical
    to NetworkOptimizationSACEnv in sac_environment.py.
    """

    def __init__(
        self,
        black_box_function,
        cc_min=1,
        cc_max=32,
        max_steps=10,
        lambda_cc=0.15,
        beta_cc=2.0,
        mu_loss=5.0,
        nu_stability=0.05,
    ):
        super().__init__()

        # The only difference from the simulator env: store black_box directly
        self.get_utility_value = black_box_function

        self.cc_min    = cc_min
        self.cc_max    = cc_max
        self.max_steps = max_steps

        # Action space: continuous [-1, 1], mapped to [cc_min, cc_max]
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([ 1.0]),
            dtype=np.float32,
        )

        # 4-D normalized observation: [u_t, c_t, Δu_t, Δc_t]
        self.observation_space = spaces.Box(
            low=np.array([ 0.0,  0.0, -1.0, -1.0]),
            high=np.array([ 1.0,  1.0,  1.0,  1.0]),
            dtype=np.float32,
        )

        # Reward hyperparameters
        self.lambda_cc    = lambda_cc
        self.beta_cc      = beta_cc
        self.mu_loss      = mu_loss
        self.nu_stability = nu_stability

        # Bandwidth estimates (set by exploration_phase before first reset)
        self.BW_est  = 1000.0
        self.TPT_est = 50.0

        # Running state
        self.current_step        = 0
        self.current_concurrency = cc_min
        self.current_throughput  = 0.0
        self.packet_loss_rate    = 0.0

        # Delta-state tracking
        self.prev_u_t = 0.0
        self.prev_c_t = 0.0

    # ------------------------------------------------------------------
    # Bandwidth estimation  (call once before the first reset)
    # ------------------------------------------------------------------

    def exploration_phase(self, duration=30, verbose=True):
        """
        Probe the real network with random concurrency values to estimate
        BW_est and TPT_est.  Identical logic to the simulator env.

        Args:
            duration: number of random probes
            verbose:  print progress
        """
        if verbose:
            print(f"[Exploration] Probing with {duration} random concurrency values...")

        throughputs   = []
        concurrencies = []

        for _ in range(duration):
            cc = np.random.randint(self.cc_min, self.cc_max + 1)

            # Exact same call signature as simulator
            throughput, _ = self.get_utility_value([cc])

            throughputs.append(throughput)
            concurrencies.append(cc)

        # Identical to simulator env
        self.BW_est  = max(throughputs)
        self.TPT_est = max(t / c for t, c in zip(throughputs, concurrencies))

        if verbose:
            cc_star = self.BW_est / (self.TPT_est + 1e-8)
            print(f"  BW_est  = {self.BW_est:.2f} Mbps")
            print(f"  TPT_est = {self.TPT_est:.4f} Mbps/thread")
            print(f"  CC*     ≈ {cc_star:.1f}  (theoretical optimum)")

        return {"BW_est": self.BW_est, "TPT_est": self.TPT_est}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, preserve_estimates=True, **kwargs):
        """
        Reset the episode counter.

        Unlike the simulator, we do NOT re-initialise the network connection.
        The network stays live; only the step counter and delta-state reset.

        preserve_estimates is always True for real-world (we keep BW_est).
        """
        self.current_step = 0

        # Random starting concurrency (same as simulator env)
        self.current_concurrency = np.random.randint(
            max(self.cc_min, 3), self.cc_max - 1
        )

        # One real measurement to initialise state
        throughput, loss = self.get_utility_value([self.current_concurrency])
        self.current_throughput = throughput
        self.packet_loss_rate   = loss

        state = self._get_normalized_state()
        self.prev_u_t = state[0]
        self.prev_c_t = state[1]

        return state

    def step(self, action):
        """
        Execute action on the real network.

        Identical to simulator env step() except it calls black_box_function
        instead of simulator.get_utility_value().
        """
        cc_actual = self._action_to_cc(action[0])

        prev_cc = self.current_concurrency
        self.current_concurrency = cc_actual

        # --- The only line different from the simulator env ---
        throughput, packet_loss = self.get_utility_value([cc_actual])
        # ------------------------------------------------------

        self.current_throughput = throughput
        self.packet_loss_rate   = packet_loss

        # EMA bandwidth update (can decrease for sudden drops)
        self.BW_est = 0.9 * self.BW_est + 0.1 * throughput
        self.BW_est = max(self.BW_est, 1.0)

        state  = self._get_normalized_state()
        reward = self._compute_reward(state, prev_cc)

        self.prev_u_t = state[0]
        self.prev_c_t = state[1]

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "throughput":  self.current_throughput,
            "concurrency": self.current_concurrency,
            "packet_loss": self.packet_loss_rate,
            "bw_est":      self.BW_est,
        }

        return state, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers  (identical to sac_environment.py)
    # ------------------------------------------------------------------

    def _action_to_cc(self, action_scalar):
        """Map continuous action in [-1, 1] to integer concurrency."""
        cc_float = (
            0.5 * (action_scalar + 1) * (self.cc_max - self.cc_min) + self.cc_min
        )
        cc_floor = int(np.floor(cc_float))
        cc_ceil  = int(np.ceil(cc_float))
        frac     = cc_float - cc_floor
        cc = cc_floor if np.random.random() < (1.0 - frac) else cc_ceil
        return int(np.clip(cc, self.cc_min, self.cc_max))

    def _get_normalized_state(self):
        """Compute 4-D scale-invariant state vector."""
        u_t     = float(np.clip(self.current_throughput / (self.BW_est + 1e-8), 0.0, 1.0))
        c_t     = self.current_concurrency / self.cc_max
        delta_u = float(np.clip(u_t - self.prev_u_t, -1.0, 1.0))
        delta_c = float(np.clip(c_t - self.prev_c_t, -1.0, 1.0))
        return np.array([u_t, c_t, delta_u, delta_c], dtype=np.float32)

    def _compute_reward(self, state, prev_cc):
        """Bounded SAC reward (identical to sac_environment.py)."""
        u_t, c_t, _, delta_c = state
        reward = (
            np.log1p(5.0 * u_t)
            - self.lambda_cc    * (c_t ** self.beta_cc)
            - self.mu_loss      * self.packet_loss_rate
            - self.nu_stability * abs(delta_c)
        )
        return float(np.clip(reward, -5.0, 5.0))
