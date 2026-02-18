import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sac_simulator import GeneralizedNetworkSimulator, ScenarioGenerator
from sac_environment import NetworkOptimizationSACEnv, ExplorationWrapper
from sac_agent import SACAgent, ReplayBuffer


class OfflineTrainer:
    """
    Offline training pipeline for SAC agent with:
    - Domain randomization
    - Curriculum learning
    - Comprehensive logging
    - Periodic evaluation
    """
    
    def __init__(
        self,
        state_dim=4,
        action_dim=1,
        hidden_size=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        buffer_capacity=1_000_000,
        batch_size=256,
        log_dir='./logs',
        checkpoint_dir='./checkpoints'
    ):
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        
        # Initialize agent
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            tau=tau,
            alpha_lr=alpha_lr
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Scenario generator (curriculum learning)
        self.scenario_generator = ScenarioGenerator()
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_interval = None  # Will be set during training
        
    def train(
        self,
        num_episodes=10000,
        updates_per_step=1,
        start_steps=1000,
        eval_interval=500,
        checkpoint_interval=1000,
        min_buffer_size=1000
    ):
        """
        Main training loop
        
        Args:
            num_episodes: total number of training episodes
            updates_per_step: how many gradient updates per environment step
            start_steps: random exploration steps before training
            eval_interval: evaluate every N episodes
            checkpoint_interval: save checkpoint every N episodes
            min_buffer_size: minimum buffer size before training
        """
        print("=" * 60)
        print("Starting SAC Offline Training")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"State dim: {self.agent.actor.fc1.in_features}")
        print(f"Action dim: {self.agent.action_dim}")
        print(f"Device: {next(self.agent.actor.parameters()).device}")
        print("=" * 60)
        
        # Store eval_interval for plotting
        self.eval_interval = eval_interval
        
        total_steps = 0
        
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            # Get scenario for this episode (curriculum learning)
            scenario = self.scenario_generator.get_scenario()
            
            # Create simulator with scenario
            simulator = GeneralizedNetworkSimulator(
                background_traffic_pattern=scenario['background_traffic_pattern'],
                enable_randomization=True
            )
            
            # Create environment
            env = NetworkOptimizationSACEnv(simulator)
            env = ExplorationWrapper(env, exploration_steps=10)
            
            # Reset
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                # Select action
                if total_steps < start_steps:
                    # Random exploration initially
                    action = env.action_space.sample()
                else:
                    # Sample from policy
                    action = self.agent.select_action(state, deterministic=False)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Train agent (if enough data)
                if len(self.replay_buffer) > min_buffer_size:
                    for _ in range(updates_per_step):
                        losses = self.agent.update(self.replay_buffer, self.batch_size)
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Increment scenario generator
            self.scenario_generator.next_episode()
            
            # Periodic evaluation
            if episode % eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=10)
                self.eval_rewards.append(eval_reward)
                
                avg_train_reward = np.mean(self.episode_rewards[-100:])
                
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Avg Train Reward (last 100): {avg_train_reward:.2f}")
                print(f"  Eval Reward: {eval_reward:.2f}")
                print(f"  Alpha: {self.agent.alpha.item():.4f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
                print(f"  Scenario: {scenario['background_traffic_pattern']}")
            
            # Save checkpoint
            if episode % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'sac_agent_episode_{episode}.pt'
                )
                self.agent.save(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_path = os.path.join(self.checkpoint_dir, 'sac_agent_final.pt')
        self.agent.save(final_path)
        print(f"\nTraining complete! Final model saved: {final_path}")
        
        # Plot results
        self.plot_training_curves()
        
    def evaluate(self, num_episodes=10):
        """
        Evaluate agent on diverse scenarios
        
        Returns:
            avg_reward: average reward over evaluation episodes
        """
        eval_rewards = []
        
        # Test on diverse scenarios - distribute episodes across patterns
        patterns = ['static', 'bursty', 'oscillating', 'sudden_drop']
        
        for i in range(num_episodes):
            # Cycle through patterns
            pattern = patterns[i % len(patterns)]
            
            # Create simulator with specific pattern
            simulator = GeneralizedNetworkSimulator(
                background_traffic_pattern=pattern,
                enable_randomization=True
            )
            
            env = NetworkOptimizationSACEnv(simulator)
            env.exploration_phase(20)  # Quick exploration
            
            state = env.reset(preserve_estimates=True)
            episode_reward = 0
            done = False
            
            while not done:
                # Deterministic action for evaluation
                action = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def plot_training_curves(self):
        """Plot training and evaluation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(
                self.episode_rewards,
                np.ones(100) / 100,
                mode='valid'
            )
            axes[0, 0].plot(smoothed, label='Smoothed (100 ep)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Evaluation rewards
        if self.eval_rewards:
            # Use actual eval_interval, not hardcoded 500
            eval_interval = self.eval_interval if self.eval_interval else 500
            eval_episodes = np.arange(len(self.eval_rewards)) * eval_interval
            axes[0, 1].plot(eval_episodes, self.eval_rewards, marker='o')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Eval Reward')
            axes[0, 1].set_title('Evaluation Performance')
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths, alpha=0.3)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].grid(True)
        
        # Reward distribution (last 1000 episodes)
        recent_rewards = self.episode_rewards[-1000:]
        axes[1, 1].hist(recent_rewards, bins=50, alpha=0.7)
        axes[1, 1].axvline(np.mean(recent_rewards), color='r', linestyle='--', label='Mean')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Reward Distribution (Last 1000 Episodes)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved: {plot_path}")
        plt.close()


def test_generalization(agent_path):
    """
    Test agent generalization across different network scales
    This demonstrates the "plug-and-play" capability
    """
    print("\n" + "=" * 60)
    print("Testing Generalization Across Network Scales")
    print("=" * 60)
    
    # Load trained agent
    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(agent_path)
    
    # Test on networks from 10 Mbps to 100 Gbps
    test_bandwidths = [10, 100, 1000, 10_000, 100_000]  # Mbps
    
    results = []
    
    for bw in test_bandwidths:
        print(f"\nTesting on {bw} Mbps network...")
        
        # Create simulator with fixed bandwidth
        simulator = GeneralizedNetworkSimulator(
            network_bandwidth=bw,
            network_throughput_per_thread=bw / 10,  # 10 threads saturate
            enable_randomization=False
        )
        
        env = NetworkOptimizationSACEnv(simulator)
        env.exploration_phase(50)  # Estimate bandwidth
        
        # Run episode - CRITICAL: preserve the estimates we just computed!
        state = env.reset(preserve_estimates=True)
        episode_reward = 0
        throughputs = []
        concurrencies = []
        
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            throughputs.append(info['throughput'])
            concurrencies.append(info['concurrency'])
        
        avg_throughput = np.mean(throughputs)
        avg_concurrency = np.mean(concurrencies)
        utilization = (avg_throughput / bw) * 100
        
        results.append({
            'bandwidth': bw,
            'reward': episode_reward,
            'throughput': avg_throughput,
            'concurrency': avg_concurrency,
            'utilization': utilization
        })
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Avg Throughput: {avg_throughput:.2f} Mbps")
        print(f"  Avg Concurrency: {avg_concurrency:.1f}")
        print(f"  Utilization: {utilization:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("Generalization Test Summary")
    print("=" * 60)
    for r in results:
        print(f"{r['bandwidth']:>10} Mbps: "
              f"Util={r['utilization']:>5.1f}%, "
              f"CC={r['concurrency']:>4.1f}, "
              f"Reward={r['reward']:>6.2f}")
    
    return results


if __name__ == "__main__":
    # Training configuration
    config = {
        'num_episodes': 10000,
        'state_dim': 4,
        'action_dim': 1,
        'hidden_size': 256,
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_capacity': 1_000_000,
        'batch_size': 256,
        'start_steps': 1000,
        'eval_interval': 500,
        'checkpoint_interval': 1000
    }
    
    # Create trainer
    trainer = OfflineTrainer(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_size=config['hidden_size'],
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size']
    )
    
    # Train
    trainer.train(
        num_episodes=config['num_episodes'],
        start_steps=config['start_steps'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval']
    )
    
    # Test generalization
    final_model_path = os.path.join(trainer.checkpoint_dir, 'sac_agent_final.pt')
    test_generalization(final_model_path)
