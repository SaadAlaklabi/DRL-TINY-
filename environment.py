"""
6G Edge Environment for DRL-TinyEdge Framework
Simulates network conditions, energy consumption, and latency dynamics
"""

import numpy as np
import random
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Optional

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class NetworkCondition(Enum):
    """Network quality states"""
    GOOD = 0    # SNR > 25 dB
    AVERAGE = 1  # 15-25 dB
    POOR = 2    # < 15 dB


class ExecutionVenue(Enum):
    """Execution locations"""
    LOCAL = 0
    EDGE = 1
    CLOUD = 2


@dataclass
class DeviceState:
    """State of edge device"""
    cpu_temp: float       # CPU temperature (°C)
    battery_level: float  # Battery percentage (0-100)
    cpu_usage: float      # Current CPU utilization (0-1)
    memory_usage: float   # Memory utilization (0-1)


@dataclass
class NetworkState:
    """State of network connection"""
    snr: float           # Signal-to-noise ratio (dB)
    rtt: float           # Round-trip time (ms)
    bandwidth: float     # Available bandwidth (Mbps)
    packet_loss: float   # Packet loss rate (0-1)


class EdgeEnvironment:
    """
    6G Edge Environment for DRL-TinyEdge
    Simulates realistic edge computing conditions
    """
    
    def __init__(self, num_devices=1, num_model_configs=5, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.num_devices = num_devices
        self.num_model_configs = num_model_configs
        
        # State space: 12 dimensions
        # [SNR, RTT, bandwidth, packet_loss, cpu_temp, battery, cpu_usage, 
        #  memory_usage, workload_type, prev_venue, prev_config, time_of_day]
        self.state_dim = 12
        
        # Action space: 3 venues × 5 model configs = 15 actions
        self.action_dim = 15
        
        # Reward coefficients (from paper)
        self.alpha = 0.35   # Latency weight
        self.beta = 0.40    # Energy weight
        self.gamma = 0.20   # Accuracy weight
        self.delta = 0.10   # Stability (switch cost) weight
        
        # SLA constraints
        self.latency_cap = 100  # ms
        self.energy_cap = 50    # mJ
        
        # Model configurations (0: ultra-lite to 4: large)
        self.model_configs = {
            0: {'name': 'Ultra-lite', 'size_kb': 15, 'base_accuracy': 0.88, 'flops': 5e6},
            1: {'name': 'Lite', 'size_kb': 35, 'base_accuracy': 0.90, 'flops': 15e6},
            2: {'name': 'Base', 'size_kb': 75, 'base_accuracy': 0.92, 'flops': 35e6},
            3: {'name': 'Medium', 'size_kb': 150, 'base_accuracy': 0.94, 'flops': 75e6},
            4: {'name': 'Large', 'size_kb': 250, 'base_accuracy': 0.95, 'flops': 150e6}
        }
        
        # Venue latency profiles (base latency in ms)
        self.venue_latency = {
            ExecutionVenue.LOCAL: {'base': 50, 'variance': 10},
            ExecutionVenue.EDGE: {'base': 30, 'variance': 15},
            ExecutionVenue.CLOUD: {'base': 20, 'variance': 25}
        }
        
        # Venue energy profiles (base energy in mJ)
        self.venue_energy = {
            ExecutionVenue.LOCAL: {'base': 35, 'variance': 5},
            ExecutionVenue.EDGE: {'base': 15, 'variance': 8},  # transmission + partial compute
            ExecutionVenue.CLOUD: {'base': 10, 'variance': 10}  # transmission only
        }
        
        # Initialize state
        self.device_state = None
        self.network_state = None
        self.current_step = 0
        self.max_steps = 2000
        self.prev_action = None
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.prev_action = None
        
        # Initialize network state (good conditions)
        self.network_state = NetworkState(
            snr=np.random.uniform(20, 30),
            rtt=np.random.uniform(10, 50),
            bandwidth=np.random.uniform(50, 100),
            packet_loss=np.random.uniform(0, 0.02)
        )
        
        # Initialize device state
        self.device_state = DeviceState(
            cpu_temp=np.random.uniform(35, 50),
            battery_level=np.random.uniform(70, 100),
            cpu_usage=np.random.uniform(0.1, 0.3),
            memory_usage=np.random.uniform(0.2, 0.4)
        )
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        # Normalize state values to [0, 1]
        state = np.array([
            self.network_state.snr / 40.0,                    # SNR normalized
            self.network_state.rtt / 200.0,                   # RTT normalized
            self.network_state.bandwidth / 200.0,             # Bandwidth normalized
            self.network_state.packet_loss,                   # Already 0-1
            self.device_state.cpu_temp / 100.0,               # Temperature normalized
            self.device_state.battery_level / 100.0,          # Battery normalized
            self.device_state.cpu_usage,                      # Already 0-1
            self.device_state.memory_usage,                   # Already 0-1
            np.random.choice([0.0, 0.5, 1.0]),               # Workload type (vision/sensor/mixed)
            (self.prev_action // 5) / 2.0 if self.prev_action else 0.5,  # Previous venue
            (self.prev_action % 5) / 4.0 if self.prev_action else 0.5,   # Previous config
            (self.current_step % 24) / 24.0                   # Time of day
        ], dtype=np.float32)
        
        return state
    
    def _update_network_conditions(self):
        """Simulate dynamic network condition changes"""
        # Random walk for network parameters
        self.network_state.snr += np.random.normal(0, 2)
        self.network_state.snr = np.clip(self.network_state.snr, 5, 35)
        
        self.network_state.rtt += np.random.normal(0, 10)
        self.network_state.rtt = np.clip(self.network_state.rtt, 5, 200)
        
        self.network_state.bandwidth += np.random.normal(0, 5)
        self.network_state.bandwidth = np.clip(self.network_state.bandwidth, 10, 150)
        
        self.network_state.packet_loss += np.random.normal(0, 0.01)
        self.network_state.packet_loss = np.clip(self.network_state.packet_loss, 0, 0.15)
        
        # Occasional network degradation events
        if random.random() < 0.05:  # 5% chance of degradation
            self.network_state.snr *= 0.7
            self.network_state.rtt *= 1.5
    
    def _update_device_state(self, action):
        """Update device state based on action"""
        venue = action // 5
        config = action % 5
        
        # Temperature increases with local computation
        if venue == 0:  # Local
            temp_increase = 0.5 * (config + 1)
            self.device_state.cpu_temp += temp_increase
        else:
            self.device_state.cpu_temp -= 0.2  # Cooling when offloading
        
        self.device_state.cpu_temp = np.clip(self.device_state.cpu_temp, 30, 90)
        
        # Battery consumption
        energy = self._compute_energy(venue, config)
        battery_drain = energy / 1000  # Convert mJ to approximate battery %
        self.device_state.battery_level -= battery_drain
        self.device_state.battery_level = np.clip(self.device_state.battery_level, 0, 100)
        
        # CPU and memory usage
        if venue == 0:
            self.device_state.cpu_usage = 0.5 + 0.1 * config
            self.device_state.memory_usage = 0.3 + 0.1 * config
        else:
            self.device_state.cpu_usage = 0.2
            self.device_state.memory_usage = 0.2
    
    def _compute_latency(self, venue: int, config: int) -> float:
        """Compute inference latency based on venue and model config"""
        venue_enum = ExecutionVenue(venue)
        venue_profile = self.venue_latency[venue_enum]
        
        # Base latency
        base = venue_profile['base']
        variance = venue_profile['variance']
        
        # Model complexity factor
        model_factor = 1 + 0.2 * config
        
        # Network condition factor
        network_factor = 1.0
        if venue != 0:  # Not local
            if self.network_state.snr < 15:
                network_factor = 2.0  # Poor network doubles latency
            elif self.network_state.snr < 25:
                network_factor = 1.3  # Average network
            
            # RTT impact
            network_factor *= (1 + self.network_state.rtt / 200)
            
            # Packet loss impact
            network_factor *= (1 + 5 * self.network_state.packet_loss)
        
        # Local thermal throttling
        if venue == 0 and self.device_state.cpu_temp > 70:
            thermal_factor = 1 + (self.device_state.cpu_temp - 70) / 30
        else:
            thermal_factor = 1.0
        
        latency = base * model_factor * network_factor * thermal_factor
        latency += np.random.normal(0, variance)
        
        return max(5, latency)  # Minimum 5ms
    
    def _compute_energy(self, venue: int, config: int) -> float:
        """Compute energy consumption based on venue and model config"""
        venue_enum = ExecutionVenue(venue)
        venue_profile = self.venue_energy[venue_enum]
        
        base = venue_profile['base']
        variance = venue_profile['variance']
        
        # Model complexity factor (larger models use more energy locally)
        if venue == 0:  # Local
            model_factor = 1 + 0.3 * config
        else:  # Offloading - energy for transmission
            model_factor = 1 + 0.1 * config
        
        # Network condition factor for offloading
        if venue != 0:
            if self.network_state.snr < 15:
                # Poor network requires more transmission power
                network_factor = 1.5
            else:
                network_factor = 1.0
        else:
            network_factor = 1.0
        
        energy = base * model_factor * network_factor
        energy += np.random.normal(0, variance)
        
        return max(5, energy)  # Minimum 5mJ
    
    def _compute_accuracy(self, venue: int, config: int) -> float:
        """Compute inference accuracy based on model config and conditions"""
        base_accuracy = self.model_configs[config]['base_accuracy']
        
        # Network impact on accuracy (for offloading)
        if venue != 0:
            if self.network_state.packet_loss > 0.1:
                accuracy_drop = 0.05
            elif self.network_state.packet_loss > 0.05:
                accuracy_drop = 0.02
            else:
                accuracy_drop = 0
        else:
            accuracy_drop = 0
        
        # Add small noise
        accuracy = base_accuracy - accuracy_drop + np.random.normal(0, 0.005)
        
        return np.clip(accuracy, 0.7, 0.99)
    
    def _compute_reward(self, latency: float, energy: float, accuracy: float,
                       switch_cost: float) -> float:
        """
        Compute multi-objective reward
        r_t = -α·Latency - β·Energy + γ·Accuracy - δ·SwitchCost
        """
        # Normalize metrics
        latency_norm = latency / self.latency_cap
        energy_norm = energy / self.energy_cap
        accuracy_norm = accuracy  # Already 0-1
        
        # Compute reward components
        latency_reward = -self.alpha * latency_norm
        energy_reward = -self.beta * energy_norm
        accuracy_reward = self.gamma * accuracy_norm
        stability_penalty = -self.delta * switch_cost
        
        reward = latency_reward + energy_reward + accuracy_reward + stability_penalty
        
        # Bonus for meeting SLA
        if latency <= self.latency_cap and energy <= self.energy_cap:
            reward += 0.1
        
        # Penalty for violating constraints
        if latency > self.latency_cap:
            reward -= 0.5 * (latency - self.latency_cap) / self.latency_cap
        if energy > self.energy_cap:
            reward -= 0.5 * (energy - self.energy_cap) / self.energy_cap
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        """
        self.current_step += 1
        
        venue = action // 5
        config = action % 5
        
        # Compute switch cost
        if self.prev_action is not None:
            prev_venue = self.prev_action // 5
            prev_config = self.prev_action % 5
            switch_cost = (venue != prev_venue) * 0.5 + (config != prev_config) * 0.3
        else:
            switch_cost = 0
        
        # Compute metrics
        latency = self._compute_latency(venue, config)
        energy = self._compute_energy(venue, config)
        accuracy = self._compute_accuracy(venue, config)
        
        # Compute reward
        reward = self._compute_reward(latency, energy, accuracy, switch_cost)
        
        # Update states
        self._update_network_conditions()
        self._update_device_state(action)
        
        self.prev_action = action
        
        # Check termination
        done = self.current_step >= self.max_steps
        
        # Episode info
        info = {
            'latency': latency,
            'energy': energy,
            'accuracy': accuracy,
            'venue': venue,
            'config': config,
            'switch_cost': switch_cost,
            'snr': self.network_state.snr,
            'rtt': self.network_state.rtt,
            'battery': self.device_state.battery_level,
            'cpu_temp': self.device_state.cpu_temp
        }
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def get_network_condition(self) -> NetworkCondition:
        """Get current network condition category"""
        if self.network_state.snr > 25:
            return NetworkCondition.GOOD
        elif self.network_state.snr > 15:
            return NetworkCondition.AVERAGE
        else:
            return NetworkCondition.POOR


class BaselineEnvironment(EdgeEnvironment):
    """
    Environment for baseline comparisons
    """
    
    def __init__(self, policy='local_only', **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
    
    def get_baseline_action(self) -> int:
        """Get action according to baseline policy"""
        
        if self.policy == 'local_only':
            # Always execute locally with medium config
            return 0 * 5 + 2  # Local venue, config 2
        
        elif self.policy == 'static_offload':
            # Always offload to cloud with large config
            return 2 * 5 + 4  # Cloud venue, config 4
        
        elif self.policy == 'heuristic_qos':
            # Rule-based: offload if RTT > 100 or SNR < 15
            if self.network_state.rtt > 100 or self.network_state.snr < 15:
                return 0 * 5 + 2  # Local, config 2
            else:
                return 2 * 5 + 3  # Cloud, config 3
        
        elif self.policy == 'tinynas_qat':
            # Fixed optimized model (config 1 with QAT)
            return 0 * 5 + 1  # Local, config 1
        
        elif self.policy == 'ofa':
            # Once-for-All: adapt config based on resources
            if self.device_state.battery_level < 30:
                config = 0  # Ultra-lite when low battery
            elif self.device_state.cpu_temp > 70:
                config = 1  # Lite when hot
            else:
                config = 2  # Base otherwise
            return 0 * 5 + config  # Always local
        
        else:
            return 0 * 5 + 2  # Default: local, config 2


if __name__ == '__main__':
    # Test environment
    env = EdgeEnvironment()
    
    print("Testing EdgeEnvironment...")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    
    # Run a few steps
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    total_reward = 0
    for i in range(10):
        action = random.randint(0, env.action_dim - 1)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action} (Venue: {info['venue']}, Config: {info['config']})")
        print(f"  Latency: {info['latency']:.2f} ms")
        print(f"  Energy: {info['energy']:.2f} mJ")
        print(f"  Accuracy: {info['accuracy']:.4f}")
        print(f"  Reward: {reward:.4f}")
    
    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
