"""
Real-World Deployment
=====================
Run your trained SAC agent on a real network transfer.

Usage:
    from deploy_realworld import run_transfer

    def my_black_box(threads):
        # your real network code
        return self.network_throughput, packet_loss_rate

    run_transfer(
        black_box_function=my_black_box,
        checkpoint_path='./checkpoints/sac_agent_final.pt',
    )
"""

import numpy as np
from sac_agent         import SACAgent, ReplayBuffer
from sac_env_realworld import RealWorldSACEnv


def run_transfer(
    black_box_function,
    checkpoint_path,
    cc_min=1,
    cc_max=32,
    max_steps=10,
    explore_steps=30,
    num_episodes=1,
    deterministic=True,
    verbose=True,
):
    """
    Run the trained agent on a real network transfer.

    Args:
        black_box_function : callable
            Must return (throughput, packet_loss_rate) for a given [thread_count].
        checkpoint_path    : str   path to .pt checkpoint
        cc_min/cc_max      : int   thread count bounds
        max_steps          : int   steps per episode
        explore_steps      : int   random probes before first episode
        num_episodes       : int   how many episodes to run
        deterministic      : bool  True = mean action (best for deployment)
        verbose            : bool  print per-step output
    """
    # 1. Build environment
    env = RealWorldSACEnv(
        black_box_function=black_box_function,
        cc_min=cc_min,
        cc_max=cc_max,
        max_steps=max_steps,
    )

    # 2. Estimate bandwidth from real probes
    env.exploration_phase(duration=explore_steps, verbose=verbose)

    # 3. Load agent
    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(checkpoint_path)

    if verbose:
        print(f"\nAgent loaded from {checkpoint_path}")

    # 4. Run episodes
    all_results = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_throughputs = []
        ep_ccs = []

        if verbose:
            print(f"\n--- Episode {ep+1}/{num_episodes} ---")
            print(f"  {'Step':<5} {'CC':<5} {'Throughput':>12}  {'Util':>6}  {'Loss':>6}  {'Reward':>8}")
            print(f"  {'-'*52}")

        done = False
        while not done:
            action = agent.select_action(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_throughputs.append(info['throughput'])
            ep_ccs.append(info['concurrency'])

            if verbose:
                util = info['throughput'] / (info['bw_est'] + 1e-8) * 100
                print(
                    f"  {env.current_step:<5} {info['concurrency']:<5} "
                    f"{info['throughput']:>12.2f}  {util:>5.1f}%  "
                    f"{info['packet_loss']:>5.3f}  {reward:>8.3f}"
                )

        result = {
            'episode':           ep + 1,
            'reward':            ep_reward,
            'avg_throughput':    np.mean(ep_throughputs),
            'avg_concurrency':   np.mean(ep_ccs),
            'utilization_pct':   np.mean(ep_throughputs) / (env.BW_est + 1e-8) * 100,
        }
        all_results.append(result)

        if verbose:
            print(f"\n  avg_throughput={result['avg_throughput']:.2f}  "
                  f"avg_cc={result['avg_concurrency']:.1f}  "
                  f"util={result['utilization_pct']:.1f}%  "
                  f"reward={ep_reward:.2f}")

    return all_results


def run_transfer_with_finetuning(
    black_box_function,
    checkpoint_path,
    cc_min=1,
    cc_max=32,
    max_steps=10,
    explore_steps=30,
    num_episodes=50,
    batch_size=64,
    save_path='./checkpoints/sac_finetuned.pt',
    verbose=True,
):
    """
    Run the agent AND update weights from real experience.
    Use this if the offline-trained agent doesn't generalize well enough
    to your specific link.
    """
    env = RealWorldSACEnv(
        black_box_function=black_box_function,
        cc_min=cc_min,
        cc_max=cc_max,
        max_steps=max_steps,
    )
    env.exploration_phase(duration=explore_steps, verbose=verbose)

    agent = SACAgent(state_dim=4, action_dim=1)
    agent.load(checkpoint_path)

    buffer = ReplayBuffer(capacity=100_000, state_dim=4, action_dim=1)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            ep_reward += reward
            state = next_state

            if len(buffer) >= batch_size:
                agent.update(buffer, batch_size)

        if verbose:
            print(f"[Finetune] ep={ep}/{num_episodes}  "
                  f"reward={ep_reward:.2f}  "
                  f"alpha={agent.alpha.item():.4f}  "
                  f"buffer={len(buffer)}")

        if ep % 10 == 0:
            agent.save(save_path)

    agent.save(save_path)
    return agent
