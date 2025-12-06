"""
DRL Agent for DRL-TinyEdge Framework
Implements Proximal Policy Optimization (PPO) for adaptive TinyML optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO algorithm.
    Actor: Outputs action probabilities
    Critic: Estimates state value
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action, action_probs, state_value


class DRLAgent(nn.Module):
    """
    Deep Reinforcement Learning Agent using PPO
    Handles joint optimization of execution venue and model configuration
    """
    
    def __init__(self, state_dim=12, action_dim=15, hidden_dim=256):
        super(DRLAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action space: 3 venues × 5 model configs = 15 actions
        # Venues: Local (0), Edge (1), Cloud (2)
        # Model configs: 0 (ultra-lite) to 4 (large)
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        
    def forward(self, state):
        return self.actor_critic(state)
    
    def get_action(self, state, deterministic=False):
        return self.actor_critic.get_action(state, deterministic)
    
    def decode_action(self, action):
        """Decode action index to (venue, model_config)"""
        venue = action // 5  # 0: Local, 1: Edge, 2: Cloud
        model_config = action % 5  # 0-4 model configurations
        return venue, model_config


class PPOMemory:
    """Experience replay buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def get_batch(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        return states, actions, rewards, values, log_probs, dones
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """
    PPO Training algorithm for DRL-TinyEdge
    Implements clipped surrogate objective with value function clipping
    """
    
    def __init__(self, agent, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5):
        
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.memory = PPOMemory()
        
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        values = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(device)
        
        return advantages, returns
    
    def update(self, next_value, epochs=10, batch_size=64):
        """Update policy using PPO algorithm"""
        
        states, actions, rewards, values, old_log_probs, dones = self.memory.get_batch()
        
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # PPO update epochs
        for _ in range(epochs):
            # Get current policy outputs
            action_probs, state_values = self.agent(states)
            dist = Categorical(action_probs)
            
            new_log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear memory
        self.memory.clear()
        
        return (total_policy_loss / epochs, 
                total_value_loss / epochs, 
                total_entropy / epochs)


class SafeExploration:
    """
    Safe exploration mechanism with action shielding
    Ensures SLA constraints are not violated during exploration
    """
    
    def __init__(self, latency_cap=100, energy_cap=50):
        self.latency_cap = latency_cap  # ms
        self.energy_cap = energy_cap     # mJ
        
    def is_safe_action(self, predicted_latency, predicted_energy):
        """Check if action is within safety bounds"""
        return predicted_latency <= self.latency_cap and predicted_energy <= self.energy_cap
    
    def get_backup_action(self):
        """Return safe backup action (local execution with mid-config)"""
        return 2  # Local venue (0) * 5 + config 2 = action 2


class EpsilonGreedyWithShield:
    """
    Epsilon-greedy exploration with safety shielding
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_rate=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.safe_explorer = SafeExploration()
        
    def select_action(self, agent, state, predicted_costs=None, deterministic=False):
        """Select action with epsilon-greedy exploration and safety shielding"""
        
        if not deterministic and random.random() < self.epsilon:
            # Random exploration
            action = torch.tensor([random.randint(0, agent.action_dim - 1)])
        else:
            # Greedy action from policy
            with torch.no_grad():
                action, _, _ = agent.get_action(state, deterministic=True)
        
        # Safety check (if cost predictions available)
        if predicted_costs is not None:
            pred_latency, pred_energy = predicted_costs
            if not self.safe_explorer.is_safe_action(pred_latency, pred_energy):
                action = torch.tensor([self.safe_explorer.get_backup_action()])
        
        return action
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)


if __name__ == '__main__':
    # Test agent
    agent = DRLAgent(state_dim=12, action_dim=15).to(device)
    
    # Test forward pass
    state = torch.randn(1, 12).to(device)
    action, probs, value = agent.get_action(state)
    
    print(f"Agent created successfully")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Sample action: {action.item()}")
    venue, config = agent.decode_action(action.item())
    print(f"Decoded: Venue={venue}, Model Config={config}")
    print(f"Value estimate: {value.item():.4f}")
