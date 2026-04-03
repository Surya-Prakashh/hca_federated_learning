import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import joblib

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
FEATURES = [
    "age",
    "vital_signs_oxygen_saturation",
    "vital_signs_diastolic_bp",
    "vital_signs_systolic_bp"
]
TARGET = "triage_level"
STATE_DIM = len(FEATURES)
ACTION_DIM = 5  # Triage levels 1, 2, 3, 4, 5 mapped to 0, 1, 2, 3, 4

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
MEMORY_SIZE = 5000
BATCH_SIZE = 64
TARGET_UPDATE = 10
EPISODES = 50 

# --- Deep Q-Network ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Environment Simulation ---
class TriageEnv:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=[TARGET] + FEATURES)
        self.df[TARGET] = self.df[TARGET].astype(int)
        
        # Simple normalization parameters
        self.means = self.df[FEATURES].mean()
        self.stds = self.df[FEATURES].std()
        
        self.current_idx = 0
        self.congestion_level = 0  # Counter to simulate hospital load

    def reset(self):
        self.current_idx = 0
        self.congestion_level = 0
        return self._get_state(self.current_idx)

    def _get_state(self, idx):
        row = self.df.iloc[idx][FEATURES].values.astype(np.float32)
        state = (row - self.means.values.astype(np.float32)) / (self.stds.values.astype(np.float32) + 1e-8)
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action_idx):
        # Action is 0-4, Triage level is 1-5
        chosen_level = action_idx + 1
        actual_level = self.df.iloc[self.current_idx][TARGET]
        
        reward = 0
        
        # Accuracy Reward
        if chosen_level == actual_level:
            reward += 10
        elif chosen_level < actual_level:
            # Over-triaging (costly but safer)
            reward -= 5
        else:
            # Under-triaging (critically dangerous)
            reward -= 20

        # Simulation of Congestion Impact
        # If the hospital is "congested" (modeled by number of high-priority assignments),
        # penalize further high-priority assignments unless absolutely necessary.
        if chosen_level <= 2:
            self.congestion_level += 1
            if self.congestion_level > 200: # Simulated threshold
                 reward -= 2 # Efficiency penalty

        self.current_idx += 1
        done = self.current_idx >= len(self.df) - 1
        next_state = self._get_state(self.current_idx) if not done else None
        
        return next_state, reward, done

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            return self.policy_net(state.unsqueeze(0)).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Standard DQN update
        current_q = self.policy_net(states).gather(1, actions)
        
        next_q = torch.zeros(BATCH_SIZE, 1)
        # Only non-terminal masks
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in next_states if s is not None])
        
        next_q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (GAMMA * next_q * (1 - dones))
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- Main Training Loop ---
if __name__ == "__main__":
    env = TriageEnv("triage_1000_records.csv")
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    
    print("Starting RL training...")
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        # Epsilon decay
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the model and normalization params
    model_data = {
        "model_state_dict": agent.policy_net.state_dict(),
        "means": env.means,
        "stds": env.stds,
        "features": FEATURES,
        "action_dim": ACTION_DIM
    }
    torch.save(model_data, "rl_triage_model.pth")
    print("RL Model saved to rl_triage_model.pth")

    # Save as pure numpy/joblib for the frontend to avoid torch dependency
    import joblib
    numpy_model = {
        "weights": {k: v.cpu().numpy() for k, v in agent.policy_net.state_dict().items()},
        "means": env.means,
        "stds": env.stds,
        "features": FEATURES,
        "action_dim": ACTION_DIM
    }
    joblib.dump(numpy_model, "rl_triage_weights.joblib")
    print("RL Model exported as rl_triage_weights.joblib (for frontend)")
