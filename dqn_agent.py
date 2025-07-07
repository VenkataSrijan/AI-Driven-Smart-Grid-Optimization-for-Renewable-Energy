import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, observation_dim, num_actions):
        super(QNetwork, self).__init__()
        # Deeper and wider architecture (from working solution)
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    # Added 'device' parameter to constructor
    def __init__(self, obs_dim, action_space, lr, gamma, eps, eps_min, eps_decay, buf_size, batch_size, device):
        self.observation_dim = obs_dim
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.action_map = {a: i for i, a in enumerate(action_space)} # Maps action value to index
        self.reverse_action_map = {i: a for i, a in enumerate(action_space)} # Maps index back to action value

        self.exploration_rate = eps
        self.min_exploration_rate = eps_min
        self.exploration_decay_rate = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device # Store the device

        # Move networks to the specified device
        self.policy_net = QNetwork(obs_dim, self.num_actions).to(device)
        self.target_net = QNetwork(obs_dim, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network should be in evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # SmoothL1Loss (Huber Loss) is generally more stable than MSELoss
        self.memory = deque(maxlen=buf_size)

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)
        else:
            # Convert numpy state to tensor and move to the specified device for inference
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad(): # No gradient calculation needed for action selection
                q_vals = self.policy_net(state_tensor)
            return self.reverse_action_map[q_vals.argmax().item()]

    def remember(self, s, a, r, s2, done):
        # Store action as its index for consistency with network output
        a_idx = self.action_map[a]
        self.memory.append((s, a_idx, r, s2, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return # Not enough samples to form a batch

        # Randomly sample a batch from replay memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack the batch (s=state, a=action, r=reward, s2=next_state, d=done_flag)
        s_batch, a_batch, r_batch, s2_batch_raw, d_batch = zip(*batch)

        # Convert numpy arrays/lists to PyTorch tensors and move to device
        s = torch.tensor(np.array(s_batch), dtype=torch.float32).to(self.device)
        a = torch.tensor(a_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        r = torch.tensor(r_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # --- FIX for UserWarning: Ensure next_states are converted efficiently ---
        # 1. Create a Python list of processed next_state numpy arrays (handling 'done' states)
        processed_s2_list_np = [
            np.zeros(self.observation_dim, dtype=np.float32) if done else next_state
            for next_state, done in zip(s2_batch_raw, d_batch)
        ]
        # 2. Convert this list of numpy arrays into a single, contiguous numpy array
        processed_s2_np = np.array(processed_s2_list_np, dtype=np.float32)
        # 3. Convert the single numpy array to a PyTorch tensor and move to device
        s2 = torch.tensor(processed_s2_np).to(self.device)
        # --- End of fix ---

        d = torch.tensor(d_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute Q-values for current states (Q(s,a))
        # .gather(1, a) selects the Q-value for the action taken in each state
        q_pred = self.policy_net(s).gather(1, a)

        # Compute max Q-value for next states (max_a' Q(s',a')) from the target network
        # .max(1)[0] gets the maximum Q-value along dimension 1 (actions)
        # .unsqueeze(1) adds a dimension to match q_pred's shape
        q_next = self.target_net(s2).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-values for the Bellman equation
        # If 'done' is true, the next state has no future reward (1 - d = 0)
        q_target = r + self.gamma * q_next * (1 - d)

        # Calculate the loss between predicted and target Q-values
        loss = self.criterion(q_pred, q_target)

        # Optimize the policy network
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        
        # Gradient clipping: clips gradients to a specified range to prevent exploding gradients
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1) # Clip gradients between -1 and 1
                
        self.optimizer.step() # Update network weights

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
        
    def update_target_network(self):
        # Update the target network's weights to match the policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())