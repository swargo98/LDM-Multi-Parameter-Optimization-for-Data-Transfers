import numpy as np
import matplotlib.pyplot as plt
import torch
from sac_agent import SACAgent
from sac_simulator import GeneralizedNetworkSimulator
from sac_environment import NetworkOptimizationSACEnv


def visualize_policy_response(agent_path, save_path='policy_response.png'):
    """
    Visualize how the trained agent responds to different network states
    """
    # Load agent
    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(agent_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Response to utilization changes
    utilizations = np.linspace(0, 1, 50)
    actions = []
    
    for u in utilizations:
        state = np.array([u, 0.5, 0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state, deterministic=True)
        # Map action [-1, 1] to concurrency [1, 32]
        cc = 0.5 * (action[0] + 1) * 31 + 1
        actions.append(cc)
    
    axes[0, 0].plot(utilizations * 100, actions, linewidth=2)
    axes[0, 0].set_xlabel('Network Utilization (%)')
    axes[0, 0].set_ylabel('Selected Concurrency')
    axes[0, 0].set_title('Agent Response to Utilization')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=16, color='r', linestyle='--', alpha=0.5, label='Mid-range')
    axes[0, 0].legend()
    
    # 2. Response to current concurrency (should agent increase/decrease?)
    concurrencies = np.linspace(0, 1, 50)
    actions_high_util = []
    actions_low_util = []
    
    for c in concurrencies:
        # High utilization (90%) - should increase CC
        state = np.array([0.9, c, 0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state, deterministic=True)
        cc = 0.5 * (action[0] + 1) * 31 + 1
        actions_high_util.append(cc)
        
        # Low utilization (30%) - should decrease CC
        state = np.array([0.3, c, 0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state, deterministic=True)
        cc = 0.5 * (action[0] + 1) * 31 + 1
        actions_low_util.append(cc)
    
    current_cc = concurrencies * 32
    axes[0, 1].plot(current_cc, actions_high_util, label='High Util (90%)', linewidth=2)
    axes[0, 1].plot(current_cc, actions_low_util, label='Low Util (30%)', linewidth=2)
    axes[0, 1].plot([0, 32], [0, 32], 'k--', alpha=0.3, label='No change')
    axes[0, 1].set_xlabel('Current Concurrency')
    axes[0, 1].set_ylabel('Selected Concurrency')
    axes[0, 1].set_title('Agent Response to Current Concurrency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Response to utilization trends (Δu)
    deltas = np.linspace(-0.5, 0.5, 50)
    actions_trending = []
    
    for delta in deltas:
        state = np.array([0.7, 0.5, delta, 0.0], dtype=np.float32)
        action = agent.select_action(state, deterministic=True)
        cc = 0.5 * (action[0] + 1) * 31 + 1
        actions_trending.append(cc)
    
    axes[1, 0].plot(deltas, actions_trending, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Utilization Trend (Δu)')
    axes[1, 0].set_ylabel('Selected Concurrency')
    axes[1, 0].set_title('Agent Response to Utilization Trends')
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True)
    
    # 4. Policy entropy across state space
    # Sample random states and compute action entropy
    states = []
    entropies = []
    
    for _ in range(1000):
        u = np.random.uniform(0, 1)
        c = np.random.uniform(0, 1)
        state = np.array([u, c, 0.0, 0.0], dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
            next(agent.actor.parameters()).device
        )
        
        with torch.no_grad():
            _, log_std = agent.actor.forward(state_tensor)
            entropy = 0.5 * (1 + np.log(2 * np.pi)) + log_std.sum()
            
        states.append([u, c])
        entropies.append(entropy.cpu().item())
    
    states = np.array(states)
    sc = axes[1, 1].scatter(states[:, 0], states[:, 1], c=entropies, 
                            cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Utilization')
    axes[1, 1].set_ylabel('Normalized Concurrency')
    axes[1, 1].set_title('Policy Entropy Across State Space')
    plt.colorbar(sc, ax=axes[1, 1], label='Entropy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Policy visualization saved: {save_path}")
    plt.close()


def test_sudden_drop_scenario(agent_path):
    """
    Test agent's ability to adapt to sudden bandwidth drops
    """
    print("\n" + "="*60)
    print("Testing Sudden Bandwidth Drop Scenario")
    print("="*60)
    
    # Load agent
    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(agent_path)
    
    # Create simulator with sudden drop
    simulator = GeneralizedNetworkSimulator(
        network_bandwidth=10_000,  # 10 Gbps
        network_throughput_per_thread=500,
        background_traffic_pattern='sudden_drop',
        enable_randomization=False
    )
    
    # Force a drop at step 5
    simulator.bandwidth_shift_time = 5
    
    env = NetworkOptimizationSACEnv(simulator)
    env.exploration_phase(50)
    
    # Run episode
    state = env.reset()
    
    history = {
        'step': [],
        'throughput': [],
        'concurrency': [],
        'utilization': [],
        'reward': []
    }
    
    done = False
    step = 0
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        
        history['step'].append(step)
        history['throughput'].append(info['throughput'])
        history['concurrency'].append(info['concurrency'])
        history['utilization'].append(state[0])
        history['reward'].append(reward)
        
        step += 1
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['step'], history['throughput'], marker='o')
    axes[0, 0].axvline(x=5, color='r', linestyle='--', label='Bandwidth Drop')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Throughput (Mbps)')
    axes[0, 0].set_title('Throughput Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['step'], history['concurrency'], marker='o', color='green')
    axes[0, 1].axvline(x=5, color='r', linestyle='--', label='Bandwidth Drop')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Concurrency')
    axes[0, 1].set_title('Agent Concurrency Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['step'], history['utilization'], marker='o', color='orange')
    axes[1, 0].axvline(x=5, color='r', linestyle='--', label='Bandwidth Drop')
    axes[1, 0].axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Target 80%')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Utilization')
    axes[1, 0].set_title('Network Utilization')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['step'], history['reward'], marker='o', color='purple')
    axes[1, 1].axvline(x=5, color='r', linestyle='--', label='Bandwidth Drop')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward Signal')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sudden_drop_test.png', dpi=150)
    print("Sudden drop test plot saved: sudden_drop_test.png")
    plt.close()
    
    # Print summary
    print("\nBehavior Summary:")
    print(f"  Before drop (steps 0-4):")
    print(f"    Avg Throughput: {np.mean(history['throughput'][:5]):.2f} Mbps")
    print(f"    Avg Concurrency: {np.mean(history['concurrency'][:5]):.1f}")
    print(f"  After drop (steps 5+):")
    print(f"    Avg Throughput: {np.mean(history['throughput'][5:]):.2f} Mbps")
    print(f"    Avg Concurrency: {np.mean(history['concurrency'][5:]):.1f}")
    
    cc_change = np.mean(history['concurrency'][5:]) - np.mean(history['concurrency'][:5])
    print(f"  Concurrency adjustment: {cc_change:.1f}")
    
    if cc_change < 0:
        print("  ✓ Agent correctly reduced concurrency after bandwidth drop")
    else:
        print("  ✗ Agent failed to adapt - concurrency should decrease")


def compare_against_static(agent_path):
    """
    Compare SAC agent against static concurrency baselines
    """
    print("\n" + "="*60)
    print("Comparing SAC vs Static Concurrency Policies")
    print("="*60)
    
    # Load agent
    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(agent_path)
    
    # Test on multiple bandwidth scenarios
    test_configs = [
        {'bw': 100, 'name': '100 Mbps'},
        {'bw': 1000, 'name': '1 Gbps'},
        {'bw': 10_000, 'name': '10 Gbps'},
        {'bw': 100_000, 'name': '100 Gbps'}
    ]
    
    static_ccs = [4, 8, 16, 32]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting on {config['name']} network...")
        
        results[config['name']] = {}
        
        # Test static policies
        for cc in static_ccs:
            simulator = GeneralizedNetworkSimulator(
                network_bandwidth=config['bw'],
                network_throughput_per_thread=config['bw'] / 10,
                enable_randomization=False
            )
            
            env = NetworkOptimizationSACEnv(simulator)
            env.exploration_phase(20)
            
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Fixed action for this concurrency
                action_value = (cc - 1) / 31 * 2 - 1  # Map to [-1, 1]
                action = np.array([action_value])
                state, reward, done, info = env.step(action)
                total_reward += reward
            
            results[config['name']][f'Static CC={cc}'] = total_reward
        
        # Test SAC agent
        simulator = GeneralizedNetworkSimulator(
            network_bandwidth=config['bw'],
            network_throughput_per_thread=config['bw'] / 10,
            enable_randomization=False
        )
        
        env = NetworkOptimizationSACEnv(simulator)
        env.exploration_phase(20)
        
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        results[config['name']]['SAC Agent'] = total_reward
    
    # Print comparison table
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for config_name, policy_results in results.items():
        print(f"\n{config_name}:")
        sorted_results = sorted(policy_results.items(), key=lambda x: x[1], reverse=True)
        for policy, reward in sorted_results:
            marker = "★" if policy == 'SAC Agent' else " "
            print(f"  {marker} {policy:20s}: {reward:>6.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization_utils.py <path_to_agent_checkpoint>")
        print("Example: python visualization_utils.py ./checkpoints/sac_agent_final.pt")
        sys.exit(1)
    
    agent_path = sys.argv[1]
    
    print("Running visualization suite...")
    
    # 1. Visualize policy responses
    visualize_policy_response(agent_path)
    
    # 2. Test sudden drop adaptation
    test_sudden_drop_scenario(agent_path)
    
    # 3. Compare against static baselines
    compare_against_static(agent_path)
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)
