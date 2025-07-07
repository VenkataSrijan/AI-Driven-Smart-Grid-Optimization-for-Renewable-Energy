import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from data_generator import DataGenerator # Ensure DataGenerator is imported if it's in a separate file
from grid_env import GridEnv
from dqn_agent import DQNAgent

# --- Set Device (Explicit GPU setup for consistency with main.py) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Set Seeds for Reproducibility ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # Add this for consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seeds set to {SEED}")

# --- Config (Match main.py and other files) ---
BATTERY_CAPACITY = 500
BATTERY_EFFICIENCY = 0.9
FUTURE_DEMAND_LOOK_AHEAD = 3
NUM_TIME_STEPS = 48
NUM_EVAL_EPISODES = 100 # Number of episodes for evaluation

# --- Corrected Model Path ---
MODEL_PATH = "dqn_grid_agent_trained.pth" # Match the save path in main.py

# Minimal agent hyperparams just for initialization (learning won't occur)
DQN_LEARNING_RATE = 0.0005 # Match main.py's value for consistency in agent init
DQN_DISCOUNT_FACTOR = 0.99 # Match main.py's value for consistency in agent init
DQN_EXPLORATION_START = 0.0 # Set to 0 for pure exploitation
DQN_EXPLORATION_MIN = 0.0 # Set to 0 for pure exploitation
DQN_EXPLORATION_DECAY = 0.0 # Not used since exploration is 0
DQN_REPLAY_BUFFER_SIZE = 1 # Minimal as replay buffer is not used for learning here
DQN_BATCH_SIZE = 1 # Minimal as batching is not used for learning here

# --- Env & Agent Init ---
# GridEnv now initializes its own DataGenerator internally
env = GridEnv(BATTERY_CAPACITY, BATTERY_EFFICIENCY, FUTURE_DEMAND_LOOK_AHEAD, NUM_TIME_STEPS)

# Pass the device to the agent constructor
agent = DQNAgent(env.observation_space_size, env.action_space,
                 DQN_LEARNING_RATE, DQN_DISCOUNT_FACTOR,
                 DQN_EXPLORATION_START, DQN_EXPLORATION_MIN, DQN_EXPLORATION_DECAY,
                 DQN_REPLAY_BUFFER_SIZE, DQN_BATCH_SIZE, device)

try:
    # Load model state dict to the correct device
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.policy_net.eval() # Set policy network to evaluation mode (turns off dropout etc.)
    print(f"✅ Model loaded from {MODEL_PATH} and set to evaluation mode.")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}. Please ensure main.py has been run to train and save the model.")
    exit()

# Explicitly ensure exploration is off for evaluation
agent.exploration_rate = 0.0
print(f"Agent exploration rate explicitly set to {agent.exploration_rate} for evaluation.")
print("--------------------------------------------------\n")

# --- Evaluation ---
eval_rewards, eval_unmet, eval_waste = [], [], []

for episode in range(NUM_EVAL_EPISODES):
    # --- FIXED: Removed explicit data generation and passing to env.reset() ---
    # The environment's reset method will now generate its own random data as designed.
    obs = env.reset() # <-- FIXED: No arguments passed here
    
    done = False
    total_reward = 0
    total_unmet = 0
    total_waste = 0

    # For plotting final episode
    battery_levels = []
    actions_taken = []
    solar = []
    wind = []
    demand = []
    grid_import = []
    grid_export = []
    unmet_ts = []
    waste_ts = []

    for t in range(NUM_TIME_STEPS):
        action_val = agent.choose_action(obs)
        action_idx = env.action_space.index(action_val)
        
        obs, reward, done, info = env.step(action_idx)

        total_reward += reward
        total_unmet += info["unmet_demand"]
        total_waste += info["wasted_energy"]

        # Only store data for the last episode for plotting
        if episode == NUM_EVAL_EPISODES - 1:
            battery_levels.append(env.current_battery_charge)
            actions_taken.append(action_val)
            solar.append(info["current_solar"])
            wind.append(info["current_wind"])
            demand.append(info["current_demand"])
            grid_import.append(info.get("grid_import", 0))
            grid_export.append(info.get("grid_export", 0))
            unmet_ts.append(info["unmet_demand"])
            waste_ts.append(info["wasted_energy"])

        if done:
            break

    eval_rewards.append(total_reward)
    eval_unmet.append(total_unmet)
    eval_waste.append(total_waste)

    print(f"Episode {episode+1}/{NUM_EVAL_EPISODES} | Reward: {total_reward:.2f} | Unmet: {total_unmet:.2f} kWh | Waste: {total_waste:.2f} kWh")

# --- Summary ---
print("\n--- Evaluation Complete ---")
print(f"📊 Avg Unmet Demand: {np.mean(eval_unmet):.2f} kWh")
print(f"📊 Avg Wasted Energy: {np.mean(eval_waste):.2f} kWh")
print(f"📊 Avg Total Reward: {np.mean(eval_rewards):.2f}")

# --- Final Episode Plot ---
print("\n📈 Plotting final episode details...")

time = range(NUM_TIME_STEPS)
plt.figure(figsize=(15, 12))

# Plot 1: Battery and Actions
plt.subplot(3, 1, 1)
plt.plot(time, battery_levels, label="Battery Level (kWh)", color="blue")
for i, act_val in enumerate(actions_taken):
    color = "green" if act_val == 1 else "red" if act_val == -1 else "gray"
    plt.scatter(i, battery_levels[i], color=color, s=40, zorder=5)
plt.title("Battery Level and Actions (Final Episode)")
plt.xlabel("Time Step")
plt.ylabel("Charge (kWh)")
plt.axhline(y=BATTERY_CAPACITY, linestyle="--", color="orange", label="Battery Capacity")
plt.axhline(y=0, linestyle="--", color="black", label="Empty Battery")
plt.grid(True)
plt.legend()

# Plot 2: Energy Flows
plt.subplot(3, 1, 2)
plt.plot(time, solar, label="Solar", color="goldenrod")
plt.plot(time, wind, label="Wind", color="skyblue")
plt.plot(time, demand, label="Demand", color="purple", linestyle="--")
plt.plot(time, grid_import, label="Grid Import", color="cyan")
plt.plot(time, grid_export, label="Grid Export", color="magenta")
plt.title("Energy Flows")
plt.xlabel("Time Step")
plt.ylabel("Energy (kWh)")
plt.grid(True)
plt.legend()

# Plot 3: Unmet and Waste
plt.subplot(3, 1, 3)
plt.plot(time, unmet_ts, label="Unmet Demand", color="red", linewidth=2)
plt.plot(time, waste_ts, label="Wasted Energy", color="orange", linewidth=2)
plt.title("Unmet Demand & Wasted Energy")
plt.xlabel("Time Step")
plt.ylabel("Energy (kWh)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("episode_detailed_performance.png")
plt.show()
print("✅ Plot saved as episode_detailed_performance.png")