import numpy as np
from queue import PriorityQueue
import random


class GeneralizedNetworkSimulator:
    """
    Enhanced simulator with domain randomization for training a 
    generalized SAC agent that works across any network.
    """
    
    def __init__(self, 
                 network_bandwidth=None,
                 network_throughput_per_thread=None,
                 background_traffic_pattern='static',
                 enable_randomization=True):
        """
        Args:
            network_bandwidth: If None, will be randomized
            network_throughput_per_thread: If None, will be randomized
            background_traffic_pattern: 'static', 'bursty', 'oscillating', 'sudden_drop'
            enable_randomization: If True, randomizes network params each episode
        """
        self.enable_randomization = enable_randomization
        self.background_traffic_pattern = background_traffic_pattern
        
        # These will be set by randomization or fixed values
        self.network_bandwidth = network_bandwidth
        self.network_throughput_per_thread = network_throughput_per_thread
        self.K = 1.002  # Concurrency penalty (kept mild for stability)
        
        # Randomization ranges (logarithmic for bandwidth)
        self.bw_range = (10, 100_000)  # 10 Mbps to 100 Gbps
        self.tpt_per_thread_range = (5, 5000)  # 5 to 5000 Mbps per thread
        
        # For simulating bandwidth fluctuations
        self.time_steps = 0
        self.bandwidth_shift_time = None
        
        # Performance tracking
        self.network_thread = 0
        self.network_throughput = 0

        if network_bandwidth is None:
            self._randomize_network_params()
        self.original_bandwidth = self.network_bandwidth
        
    def reset(self, seed=None):
        """Reset and potentially randomize the simulator"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.enable_randomization:
            self._randomize_network_params()
        
        self.time_steps = 0
        self.network_thread = 0
        self.network_throughput = 0
        
        # Store original for sudden drop scenarios
        self.original_bandwidth = self.network_bandwidth
        
        # Randomly decide if/when to drop bandwidth (20% of episodes)
        if self.background_traffic_pattern == 'sudden_drop' and random.random() < 0.2:
            # Drop will occur between 30-70% of episode
            self.bandwidth_shift_time = random.randint(3, 7)
        else:
            self.bandwidth_shift_time = None
            
        return {
            'bw_est': self.network_bandwidth,
            'tpt_est': self.network_throughput_per_thread,
            'max_threads': 32  # Fixed for simplicity
        }
    
    def _randomize_network_params(self):
        """
        Randomize network parameters using logarithmic distribution
        This ensures we train on 1 Gbps and 100 Gbps with equal probability
        """
        # Logarithmic sampling for bandwidth (equal density on log scale)
        log_bw_min = np.log10(self.bw_range[0])
        log_bw_max = np.log10(self.bw_range[1])
        log_bw = np.random.uniform(log_bw_min, log_bw_max)
        self.network_bandwidth = 10 ** log_bw
        
        # Logarithmic sampling for per-thread throughput
        log_tpt_min = np.log10(self.tpt_per_thread_range[0])
        log_tpt_max = np.log10(self.tpt_per_thread_range[1])
        log_tpt = np.random.uniform(log_tpt_min, log_tpt_max)
        self.network_throughput_per_thread = 10 ** log_tpt
        
        # Ensure tpt_per_thread doesn't exceed bandwidth
        self.network_throughput_per_thread = min(
            self.network_throughput_per_thread,
            self.network_bandwidth * 0.9  # Max 90% of BW per thread
        )
        
        # Randomly select background traffic pattern
        patterns = ['static', 'bursty', 'oscillating', 'sudden_drop']
        self.background_traffic_pattern = random.choice(patterns)
    
    def _apply_background_traffic(self, time_step):
        """
        Simulate different background traffic patterns
        Returns: effective_bandwidth (reduced from nominal)
        """
        if self.background_traffic_pattern == 'static':
            # No variation
            return self.network_bandwidth
        
        elif self.background_traffic_pattern == 'bursty':
            # Random bursts reduce bandwidth by 20-60%
            if random.random() < 0.1:  # 10% chance of burst
                reduction = random.uniform(0.2, 0.6)
                return self.network_bandwidth * (1 - reduction)
            return self.network_bandwidth
        
        elif self.background_traffic_pattern == 'oscillating':
            # Sinusoidal variation ±30%
            variation = 0.3 * np.sin(time_step * 0.5)
            return self.network_bandwidth * (1 + variation)
        
        elif self.background_traffic_pattern == 'sudden_drop':
            # Sudden drop scenario (teaches agent to adapt)
            if self.bandwidth_shift_time and time_step >= self.bandwidth_shift_time:
                # Drop to 30-70% of original
                drop_factor = random.uniform(0.3, 0.7)
                return self.original_bandwidth * drop_factor
            return self.network_bandwidth
        
        return self.network_bandwidth
    
    def _simulate_network_thread(self, remaining_bw_per_thread):
        """Simulate a single thread's contribution"""
        # Random file size with heavy-tailed distribution
        if random.random() < 0.7:
            file_size = random.uniform(1, 100)  # Small files
        else:
            file_size = random.uniform(100, 1000)  # Large files
        
        # Throughput is limited by per-thread cap and remaining bandwidth
        throughput = min(file_size, remaining_bw_per_thread)
        
        # Add realistic noise (±10%)
        noise = random.uniform(0.9, 1.1)
        throughput *= noise
        
        return throughput
    
    def get_utility_value(self, threads):
        """
        Simulate transfer with given thread count and return utility
        
        Args:
            threads: array [network_thread_count]
        
        Returns:
            throughput: achieved throughput (Mbps)
            cpu_overhead: simulated CPU overhead
        """
        network_thread = int(threads[0])
        self.network_thread = network_thread
        
        # Apply background traffic effects
        effective_bandwidth = self._apply_background_traffic(self.time_steps)
        self.time_steps += 1
        
        # Initialize simulation
        thread_queue = PriorityQueue()
        self.network_throughput = 0
        
        # Each thread gets equal share of bandwidth initially
        per_thread_bandwidth = min(
            self.network_throughput_per_thread,
            effective_bandwidth / max(network_thread, 1)
        )
        
        # Queue all threads
        for i in range(network_thread):
            thread_queue.put((0, per_thread_bandwidth))
        
        # Simulate transfer for 1 time unit
        simulation_duration = 1.0
        network_thread_finish_time = 0
        
        while not thread_queue.empty():
            time, remaining_bw = thread_queue.get()
            
            if time >= simulation_duration:
                break
            
            # Simulate this thread's work
            throughput_increment = self._simulate_network_thread(remaining_bw)
            throughput_increment = min(
                throughput_increment,
                effective_bandwidth - self.network_throughput
            )
            
            self.network_throughput += throughput_increment
            remaining_bw -= throughput_increment
            
            # Calculate time taken
            time_taken = throughput_increment / (self.network_throughput_per_thread + 1e-8)
            next_time = time + time_taken + 0.01
            
            if next_time < simulation_duration and remaining_bw > 0:
                thread_queue.put((next_time, remaining_bw))
            
            network_thread_finish_time = max(network_thread_finish_time, next_time)
        
        # Normalize by time
        if network_thread_finish_time > 0:
            self.network_throughput = self.network_throughput / network_thread_finish_time
        
        # Simulate CPU overhead (quadratic with thread count)
        # At high thread counts, this reduces effective throughput
        cpu_overhead_factor = 1.0 - min(0.3, (network_thread / 32) ** 2 * 0.3)
        self.network_throughput *= cpu_overhead_factor
        
        # Simulated packet loss (increases with utilization)
        utilization = self.network_throughput / (effective_bandwidth + 1e-8)
        packet_loss_rate = 0.0
        if utilization > 0.8:
            packet_loss_rate = (utilization - 0.8) * 0.1  # Up to 2% loss at 100% util
        
        return self.network_throughput, packet_loss_rate
    
    def get_state_info(self):
        """Return current state for normalization"""
        return {
            'throughput': self.network_throughput,
            'concurrency': self.network_thread,
            'bw_est': self.network_bandwidth,
            'packet_loss': 0.0  # Could be enhanced
        }


class ScenarioGenerator:
    """
    Generates training scenarios with curriculum learning
    Start with simple scenarios, gradually increase difficulty
    """
    
    def __init__(self):
        self.episode_count = 0
        
    def get_scenario(self):
        """
        Return scenario difficulty based on training progress
        """
        # First 1000 episodes: Simple static networks
        if self.episode_count < 1000:
            return {
                'background_traffic_pattern': 'static',
                'bandwidth_stability': 1.0
            }
        
        # Next 2000 episodes: Introduce mild variations
        elif self.episode_count < 3000:
            pattern = random.choice(['static', 'static', 'bursty'])
            return {
                'background_traffic_pattern': pattern,
                'bandwidth_stability': 0.9
            }
        
        # Next 3000 episodes: Full diversity
        elif self.episode_count < 6000:
            pattern = random.choice(['static', 'bursty', 'oscillating'])
            return {
                'background_traffic_pattern': pattern,
                'bandwidth_stability': 0.8
            }
        
        # After 6000: Include sudden drops (hardest)
        else:
            pattern = random.choice(['static', 'bursty', 'oscillating', 'sudden_drop'])
            return {
                'background_traffic_pattern': pattern,
                'bandwidth_stability': 0.7
            }
    
    def next_episode(self):
        self.episode_count += 1
