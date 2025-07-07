import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from grid_env import GridEnv
from dqn_agent import DQNAgent

# --- Configuration ---
BATTERY_CAPACITY = 500
BATTERY_EFFICIENCY = 0.9
FUTURE_DEMAND_LOOK_AHEAD = 3 # Kept as per working solution
NUM_TIME_STEPS = 48 # Number of 1-hour intervals in an episode (e.g., 2 days for 24-hour steps)

# --- RL Agent Parameters (Tuned Values from working solution) ---
DQN_LEARNING_RATE = 0.0005
DQN_DISCOUNT_FACTOR = 0.99
DQN_EXPLORATION_START = 1.0
DQN_EXPLORATION_MIN = 0.01
DQN_EXPLORATION_DECAY = 0.998 # Note: Logs show this effectively drops to MIN very fast.

DQN_REPLAY_BUFFER_SIZE = 50000
DQN_BATCH_SIZE = 64

# --- Training Parameters ---
NUM_EPISODES = 1000 # Kept as per working solution (agent converges well within this)
TARGET_UPDATE_FREQUENCY = 10
SAVE_MODEL_FREQUENCY = 100

MODEL_SAVE_PATH = "dqn_grid_agent_trained.pth"

# --- Set Device (Explicit GPU setup for efficiency) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Set Seeds for Reproducibility ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seeds set to {SEED}")


def train_agent():
    env = GridEnv(BATTERY_CAPACITY, BATTERY_EFFICIENCY, FUTURE_DEMAND_LOOK_AHEAD, NUM_TIME_STEPS)

    observation_space_size = env.observation_space_size
    action_space_values = env.action_space

    agent = DQNAgent(
        observation_space_size,
        action_space_values,
        DQN_LEARNING_RATE,
        DQN_DISCOUNT_FACTOR,
        DQN_EXPLORATION_START,
        DQN_EXPLORATION_MIN,
        DQN_EXPLORATION_DECAY,
        DQN_REPLAY_BUFFER_SIZE,
        DQN_BATCH_SIZE,
        device # Pass the detected device to the agent
    )

    print(f"\nStarting training for {NUM_EPISODES} episodes...")
    print(f"Observation space size: {observation_space_size}")
    print(f"Action space (values): {action_space_values}")

    total_rewards = []
    unmet_history = []
    waste_history = []
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}...")
        # Load model state dict to the correct device
        agent.policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        agent.target_net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Model loaded. Training will continue from this point.")
    else:
        print("No existing model found. Starting training from scratch.")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_unmet = 0
        ep_waste = 0

        for t in range(NUM_TIME_STEPS):
            # Ensure state is handled on the correct device for action selection
            action_val = agent.choose_action(state)
            
            # Convert action_val back to index for environment step
            action_idx = action_space_values.index(action_val) 
            
            next_state, reward, done, info = env.step(action_idx)
            
            # Store experience (states are typically numpy arrays here, converted to tensors on device in learn)
            agent.remember(state, action_val, reward, next_state, done)
            
            state = next_state
            ep_reward += reward
            ep_unmet += info["unmet_demand"]
            ep_waste += info["wasted_energy"]
            
            # Check for sufficient memory size before learning
            if len(agent.memory) > DQN_BATCH_SIZE: # Corrected variable name from DQN_BATCH_BATCH_SIZE
                agent.learn()
            
            if done:
                break

        total_rewards.append(ep_reward)
        unmet_history.append(ep_unmet)
        waste_history.append(ep_waste)

        if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

        if (episode + 1) % SAVE_MODEL_FREQUENCY == 0:
            torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
            print(f"  Model saved to {MODEL_SAVE_PATH}")

        # Logging progress every 100 episodes or at the end
        if (episode + 1) % 100 == 0 or episode == NUM_EPISODES - 1:
            avg_reward_last_100 = np.mean(total_rewards[-100:])
            avg_unmet_last_100 = np.mean(unmet_history[-100:])
            avg_waste_last_100 = np.mean(waste_history[-100:])
            
            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
            print(f"  Total Reward: {ep_reward:.2f}")
            print(f"  Exploration Rate: {agent.exploration_rate:.4f}")
            print(f"  Unmet Demand (current episode): {ep_unmet:.2f} kWh")
            print(f"  Wasted Energy (current episode): {ep_waste:.2f} kWh")
            print(f"  Avg Reward (last 100): {avg_reward_last_100:.2f}")
            print(f"  Avg Unmet Demand (last 100): {avg_unmet_last_100:.2f} kWh")
            print(f"  Avg Wasted Energy (last 100): {avg_waste_last_100:.2f} kWh")

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Final model saved to {MODEL_SAVE_PATH}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(total_rewards, label="Episode Reward", color="blue", alpha=0.7)
    plt.title("Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(unmet_history, label="Unmet Demand (kWh)", color="red", alpha=0.7)
    plt.plot(waste_history, label="Wasted Energy (kWh)", color="orange", alpha=0.7)
    plt.title("Unmet Demand & Wasted Energy Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Energy (kWh)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_agent()