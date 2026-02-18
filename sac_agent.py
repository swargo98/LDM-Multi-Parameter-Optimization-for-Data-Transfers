import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN, LOG_STD_MAX = -20, 2


class Actor(nn.Module):
    """
    Stochastic policy network for SAC
    Outputs mean and log_std for a Gaussian distribution
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        
        # Input embedding
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Residual blocks (inspired by your existing architecture)
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Output heads
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        """Forward pass - returns mean and log_std"""
        x = F.relu(self.ln1(self.fc1(state)))
        
        # Residual connections
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)
        
        residual = x
        x = self.res_block2(x)
        x = F.relu(x + residual)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from policy with reparameterization trick
        Returns: action, log_prob, mean_action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        z = normal.rsample()  # Sample with gradient
        
        # Squash to [-1, 1] with tanh
        action = torch.tanh(z)
        
        # Compute log probability with tanh correction
        # log π(a|s) = log N(z|μ,σ) - log(1 - tanh²(z))
        log_prob = normal.log_prob(z)
        
        # Tanh correction (important for entropy calculation)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Mean action (for deterministic evaluation)
        mean_action = torch.tanh(mean)
        
        return action, log_prob, mean_action


class Critic(nn.Module):
    """
    Q-network for SAC
    Takes state and action, outputs Q-value
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        
        # Input is concatenated [state, action]
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        q_value = self.fc_out(x)
        return q_value


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning
    """
    
    def __init__(self, capacity=1_000_000, state_dim=None, action_dim=None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        # Validate shapes if dimensions are provided
        if self.state_dim is not None:
            assert state.shape == (self.state_dim,), f"State shape {state.shape} != {(self.state_dim,)}"
            assert next_state.shape == (self.state_dim,), f"Next state shape {next_state.shape} != {(self.state_dim,)}"
        
        if self.action_dim is not None:
            assert action.shape == (self.action_dim,), f"Action shape {action.shape} != {(self.action_dim,)}"
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample random batch"""
        # Check for empty buffer
        if len(self.buffer) == 0:
            raise ValueError("ReplayBuffer is empty, cannot sample")
        
        # Safety: if buffer is smaller than batch_size, sample with replacement
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent with automatic temperature tuning
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        auto_entropy_tuning=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Twin Critics
        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Target Critics (for stability)
        self.critic1_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_size).to(device)
        
        # Initialize targets to match main networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Automatic temperature tuning (CRITICAL for generalization)
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy = -dim(A) (heuristic from SAC paper)
            self.target_entropy = -float(action_dim)
            self.alpha_lr = alpha_lr  # Store for checkpointing
            
            # log(alpha) is optimized (ensures alpha > 0)
            # Use nn.Parameter so it's properly tracked
            self.log_alpha = torch.nn.Parameter(
                torch.zeros(1, device=device)
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.2).to(device)
    
    def select_action(self, state, deterministic=False):
        """
        Select action from policy
        
        Args:
            state: current state (numpy array)
            deterministic: if True, use mean action (for evaluation)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        """
        Perform one SAC update step
        
        Returns:
            dict: losses for logging
        """
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # ==================== Update Critics ====================
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
            # Compute target Q-values (min of two critics)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term (key to SAC)
            target_q = target_q - self.alpha * next_log_prob
            
            # Compute target
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        # Critic losses (MSE)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ==================== Update Actor ====================
        # Sample new actions from current policy
        new_action, log_prob, _ = self.actor.sample(state)
        
        # Q-values for new actions
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q - α*log_prob (equivalent to maximize expected return + entropy)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== Update Temperature (Alpha) ====================
        alpha_loss = None
        if self.auto_entropy_tuning:
            # α should increase if entropy is below target, decrease if above
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # ==================== Update Target Networks ====================
        # Soft update: θ_target = τ*θ + (1-τ)*θ_target
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        # Return losses for logging
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0
        }
    
    def _soft_update(self, source, target):
        """Soft update: target = τ*source + (1-τ)*target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            # Save config for easier loading
            'config': {
                'state_dim': self.actor.fc1.in_features,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'auto_entropy_tuning': self.auto_entropy_tuning,
                'target_entropy': self.target_entropy if self.auto_entropy_tuning else None,
            }
        }
        
        # Handle temperature parameters based on tuning mode
        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha.item()  # Save as scalar
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            # Save actual learning rate from optimizer, not hardcoded value
            checkpoint['config']['alpha_lr'] = self.alpha_optimizer.param_groups[0]['lr']
        else:
            checkpoint['alpha'] = self.alpha.item()
            checkpoint['alpha_optimizer'] = None
            checkpoint['config']['alpha_lr'] = None
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load network weights
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        
        # Restore config
        config = checkpoint.get('config', {})
        saved_auto_tuning = config.get('auto_entropy_tuning', True)
        
        # Warn if auto_entropy_tuning mismatch
        if saved_auto_tuning != self.auto_entropy_tuning:
            print(f"WARNING: Checkpoint was trained with auto_entropy_tuning={saved_auto_tuning}, "
                  f"but agent was created with {self.auto_entropy_tuning}. "
                  f"Using checkpoint's setting.")
            self.auto_entropy_tuning = saved_auto_tuning
        
        # Restore temperature tuning parameters
        if self.auto_entropy_tuning and checkpoint.get('alpha_optimizer') is not None:
            # Restore target entropy
            if 'target_entropy' in config and config['target_entropy'] is not None:
                self.target_entropy = config['target_entropy']
            
            # Recreate log_alpha as nn.Parameter
            self.log_alpha = torch.nn.Parameter(
                torch.tensor([checkpoint['log_alpha']], device=device)
            )
            self.alpha = self.log_alpha.exp()
            
            # Recreate optimizer with saved learning rate
            alpha_lr = config.get('alpha_lr', 3e-4)
            self.alpha_lr = alpha_lr
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            
            # Load optimizer state
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            
            # Move alpha optimizer state tensors to correct device
            self._move_optimizer_to_device(self.alpha_optimizer)
            
        elif not self.auto_entropy_tuning and 'alpha' in checkpoint:
            # Fixed alpha mode
            self.alpha = torch.tensor(checkpoint['alpha']).to(device)
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        # CRITICAL: Move all optimizer states to correct device
        self._move_optimizer_to_device(self.actor_optimizer)
        self._move_optimizer_to_device(self.critic1_optimizer)
        self._move_optimizer_to_device(self.critic2_optimizer)
    
    def _move_optimizer_to_device(self, optimizer):
        """Move optimizer state tensors to the correct device"""
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
