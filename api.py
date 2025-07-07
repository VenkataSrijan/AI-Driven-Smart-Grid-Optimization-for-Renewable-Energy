import torch
import numpy as np
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Assuming these are in the same directory or accessible via PYTHONPATH
from grid_env import GridEnv
from dqn_agent import DQNAgent

# --- Configuration (Ensure these match your main.py/evaluate_agent.py configs) ---
BATTERY_CAPACITY = 500
BATTERY_EFFICIENCY = 0.9
FUTURE_DEMAND_LOOK_AHEAD = 3
NUM_TIME_STEPS = 48

# DQN Agent Hyperparameters (used for agent initialization, actual learning won't happen here)
DQN_LEARNING_RATE = 0.0005
DQN_DISCOUNT_FACTOR = 0.99
DQN_EXPLORATION_START = 0.0  # Set to 0.0 for pure exploitation in API
DQN_EXPLORATION_MIN = 0.0
DQN_EXPLORATION_DECAY = 0.0

DQN_REPLAY_BUFFER_SIZE = 1 # Minimal, as replay buffer not used for API
DQN_BATCH_SIZE = 1 # Minimal, as batching not used for API

MODEL_SAVE_PATH = "dqn_grid_agent_trained.pth"

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FastAPI App Setup ---
app = FastAPI(
    title="Smart Grid AI Agent API",
    description="API for controlling and simulating a smart grid with a trained DQN agent.",
    version="1.0.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Agent and Environment State ---
ai_agent: DQNAgent = None
grid_environment_live: GridEnv = None # Use a distinct name for the live simulation env
current_observation_live: np.ndarray = None
current_episode_step_live: int = 0
episode_history_live = [] # To store data for plotting/analysis of the current API-driven episode

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Request/Response ---
class AgentActionResponse(BaseModel):
    reward: float
    battery_charge: float
    current_solar: float
    current_wind: float
    current_demand: float
    grid_import: float
    grid_export: float
    unmet_demand: float
    wasted_energy: float
    is_done: bool
    episode_step: int

class SingleActionInput(BaseModel):
    battery_charge: float
    net_energy: float
    avg_future_net_energy: float
    time_step: int

class SingleActionPrediction(BaseModel):
    action: int # Action value (-1, 0, 1)
    action_description: str
    predicted_q_values: dict # Q-values for each possible action

class SimulationStepData(BaseModel):
    time_step: int
    reward: float
    battery_charge: float
    solar_generation: float
    wind_generation: float
    total_demand: float # Renamed from 'demand' for clarity in frontend
    grid_import: float
    grid_export: float
    unmet_demand: float
    wasted_energy: float
    action_taken: int

# --- API Endpoints ---

@app.on_event("startup")
async def load_model():
    """
    Loads the AI model and initializes the environment when the FastAPI app starts.
    """
    global ai_agent, grid_environment_live, current_observation_live, current_episode_step_live
    logger.info("Loading AI model and initializing environment...")

    try:
        # Initialize a live environment for step-by-step control
        grid_environment_live = GridEnv(BATTERY_CAPACITY, BATTERY_EFFICIENCY, FUTURE_DEMAND_LOOK_AHEAD, NUM_TIME_STEPS)
        logger.debug(f"grid_environment_live.observation_space_size = {grid_environment_live.observation_space_size}")
        logger.debug(f"grid_environment_live.action_space = {grid_environment_live.action_space}")

        # Initialize the agent
        ai_agent = DQNAgent(
            grid_environment_live.observation_space_size,
            grid_environment_live.action_space,
            DQN_LEARNING_RATE,
            DQN_DISCOUNT_FACTOR,
            DQN_EXPLORATION_START,
            DQN_EXPLORATION_MIN,
            DQN_EXPLORATION_DECAY,
            DQN_REPLAY_BUFFER_SIZE,
            DQN_BATCH_SIZE,
            device # Pass the device here
        )

        # Load the trained model's state dictionary
        ai_agent.policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        ai_agent.policy_net.eval()  # Set policy network to evaluation mode
        ai_agent.exploration_rate = 0.0 # Ensure no exploration for API usage
        logger.info(f"✅ Model loaded from {MODEL_SAVE_PATH} successfully.")

        # Reset the environment for the first live episode
        current_observation_live = grid_environment_live.reset()
        current_episode_step_live = 0
        episode_history_live.clear()
        logger.info("Live environment reset for first API interaction.")

    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_SAVE_PATH}. Please ensure you have trained the model using main.py.")
        raise HTTPException(status_code=500, detail=f"Model file not found at {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}")
        raise e # Re-raise the exception to fail startup

@app.post("/api/predict_action", response_model=SingleActionPrediction)
async def predict_action(input_data: SingleActionInput):
    """
    Predicts the best action for a given state without advancing the simulation.
    Uses the agent's policy network to get Q-values.
    """
    if ai_agent is None:
        raise HTTPException(status_code=503, detail="AI agent not initialized. Please wait for server startup.")

    try:
        # Create a scaled observation from the input data
        # Note: We need a temporary env instance to use its _scale method
        temp_env = GridEnv(BATTERY_CAPACITY, BATTERY_EFFICIENCY, FUTURE_DEMAND_LOOK_AHEAD, NUM_TIME_STEPS)
        
        # Manually create the observation array in the same order as _scale
        # This requires careful mapping from frontend inputs to the observation space.
        # Ensure the scaling factors match grid_env.py
        scaled_battery_charge = input_data.battery_charge / BATTERY_CAPACITY
        scaled_net_energy = (input_data.net_energy + 200) / 400 # Assuming same scaling as GridEnv
        scaled_avg_future_net_energy = (input_data.avg_future_net_energy + 200) / 400 # Assuming same scaling
        scaled_time_step = input_data.time_step / (NUM_TIME_STEPS - 1)

        observation_from_input = np.array([
            scaled_battery_charge,
            scaled_net_energy,
            scaled_avg_future_net_energy,
            scaled_time_step
        ], dtype=np.float32)

        # Get Q-values from the policy network
        state_tensor = torch.tensor(observation_from_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values_tensor = ai_agent.policy_net(state_tensor)
        
        predicted_action_idx = q_values_tensor.argmax().item()
        predicted_action_val = ai_agent.reverse_action_map[predicted_action_idx]

        q_values_dict = {
            ai_agent.reverse_action_map[i]: q_values_tensor[0][i].item()
            for i in range(ai_agent.num_actions)
        }

        action_description = ""
        if predicted_action_val == 1:
            action_description = "Charge Battery / Export to Grid"
        elif predicted_action_val == -1:
            action_description = "Discharge Battery / Import from Grid"
        else: # predicted_action_val == 0
            action_description = "Maintain Balance / Do Nothing"

        logger.info(f"Predicted action for input state: {predicted_action_val}")

        return SingleActionPrediction(
            action=predicted_action_val,
            action_description=action_description,
            predicted_q_values=q_values_dict
        )
    except Exception as e:
        logger.error(f"Error predicting action: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting action: {e}")

@app.post("/api/simulate_episode", response_model=list[SimulationStepData])
async def simulate_episode():
    """
    Runs a full 48-step simulation and returns the complete history.
    This uses a new, temporary environment instance for each full simulation.
    """
    if ai_agent is None:
        raise HTTPException(status_code=503, detail="AI agent not initialized. Please wait for server startup.")

    logger.info("Running full episode simulation...")
    simulation_env = GridEnv(BATTERY_CAPACITY, BATTERY_EFFICIENCY, FUTURE_DEMAND_LOOK_AHEAD, NUM_TIME_STEPS)
    state = simulation_env.reset()
    simulation_history = []

    for t in range(NUM_TIME_STEPS):
        action_val = ai_agent.choose_action(state)
        action_idx = simulation_env.action_space.index(action_val)
        next_state, reward, done, info = simulation_env.step(action_idx)
        
        # Store data for this step
        simulation_history.append(SimulationStepData(
            time_step=t,
            reward=reward,
            battery_charge=simulation_env.current_battery_charge,
            solar_generation=info["current_solar"],
            wind_generation=info["current_wind"],
            total_demand=info["current_demand"],
            grid_import=info["grid_import"],
            grid_export=info["grid_export"],
            unmet_demand=info["unmet_demand"],
            wasted_energy=info["wasted_energy"],
            action_taken=action_val
        ))
        state = next_state
        if done:
            break
    
    logger.info(f"Full episode simulation complete. {len(simulation_history)} steps recorded.")
    return simulation_history

@app.get("/api/reset_env", response_model=AgentActionResponse)
async def reset_live_environment():
    """
    Resets the LIVE simulation environment to a new random episode.
    """
    global current_observation_live, current_episode_step_live, episode_history_live
    if grid_environment_live is None:
        raise HTTPException(status_code=503, detail="AI agent and environment not initialized. Please wait for server startup.")

    logger.info("Resetting LIVE environment for a new episode...")
    current_observation_live = grid_environment_live.reset()
    current_episode_step_live = 0
    episode_history_live.clear()

    initial_info = {
        "reward": 0.0, # Initial reward is 0
        "battery_charge": grid_environment_live.current_battery_charge,
        "current_solar": grid_environment_live.solar[0],
        "current_wind": grid_environment_live.wind[0],
        "current_demand": grid_environment_live.demand[0],
        "grid_import": 0.0, # Assumed 0 at initial reset
        "grid_export": 0.0, # Assumed 0 at initial reset
        "unmet_demand": 0.0, # Assumed 0 at initial reset
        "wasted_energy": 0.0, # Assumed 0 at initial reset
        "is_done": False,
        "episode_step": current_episode_step_live
    }
    episode_history_live.append(initial_info) # Store initial state

    logger.info("LIVE environment reset complete.")
    return AgentActionResponse(**initial_info)

@app.get("/api/take_action", response_model=AgentActionResponse)
async def take_live_action():
    """
    Instructs the AI agent to take one action in the current LIVE environment state.
    """
    global current_observation_live, current_episode_step_live, episode_history_live
    if ai_agent is None or grid_environment_live is None:
        raise HTTPException(status_code=503, detail="AI agent and environment not initialized. Please wait for server startup.")
    
    if current_episode_step_live >= NUM_TIME_STEPS:
        logger.warning("LIVE episode is already done. Please reset the environment via /api/reset_env.")
        # Return the last state if already done
        return AgentActionResponse(
            reward=0.0, # Or some indicator
            battery_charge=grid_environment_live.current_battery_charge,
            current_solar=grid_environment_live.solar[-1], # Last known data
            current_wind=grid_environment_live.wind[-1],
            current_demand=grid_environment_live.demand[-1],
            grid_import=0.0, grid_export=0.0, unmet_demand=0.0, wasted_energy=0.0,
            is_done=True, episode_step=current_episode_step_live
        )

    logger.info(f"Taking action at LIVE episode step {current_episode_step_live}...")
    
    # Agent decides action based on current observation
    action_val = ai_agent.choose_action(current_observation_live)
    action_idx = grid_environment_live.action_space.index(action_val)

    # Environment steps forward
    next_obs, reward, done, info = grid_environment_live.step(action_idx)
    current_observation_live = next_obs # Update observation for the next step
    current_episode_step_live += 1

    # Prepare response data
    response_data = {
        "reward": reward,
        "battery_charge": grid_environment_live.current_battery_charge,
        "current_solar": info["current_solar"],
        "current_wind": info["current_wind"],
        "current_demand": info["current_demand"],
        "grid_import": info["grid_import"],
        "grid_export": info["grid_export"],
        "unmet_demand": info["unmet_demand"],
        "wasted_energy": info["wasted_energy"],
        "is_done": done,
        "episode_step": current_episode_step_live
    }
    episode_history_live.append(response_data) # Store for history endpoint

    logger.info(f"Action taken. Step: {current_episode_step_live}, Reward: {reward:.2f}, Battery: {grid_environment_live.current_battery_charge:.2f}")
    return AgentActionResponse(**response_data)

@app.get("/api/episode_history", response_model=list[AgentActionResponse])
async def get_live_episode_history():
    """
    Returns the historical data for the current LIVE simulation episode.
    """
    if ai_agent is None or grid_environment_live is None:
        raise HTTPException(status_code=503, detail="AI agent and environment not initialized.")
    
    return episode_history_live

@app.get("/api/current_status", response_model=AgentActionResponse)
async def get_live_current_status():
    """
    Returns the current status of the LIVE environment without taking an action.
    """
    if ai_agent is None or grid_environment_live is None:
        raise HTTPException(status_code=503, detail="AI agent and environment not initialized. Please wait for server startup.")
    
    if not episode_history_live:
        # If history is empty, return initial state from the live environment's current state
        initial_info = {
            "reward": 0.0,
            "battery_charge": grid_environment_live.current_battery_charge,
            "current_solar": grid_environment_live.solar[current_episode_step_live] if current_episode_step_live < NUM_TIME_STEPS else 0.0,
            "current_wind": grid_environment_live.wind[current_episode_step_live] if current_episode_step_live < NUM_TIME_STEPS else 0.0,
            "current_demand": grid_environment_live.demand[current_episode_step_live] if current_episode_step_live < NUM_TIME_STEPS else 0.0,
            "grid_import": 0.0,
            "grid_export": 0.0,
            "unmet_demand": 0.0,
            "wasted_energy": 0.0,
            "is_done": (current_episode_step_live >= NUM_TIME_STEPS),
            "episode_step": current_episode_step_live
        }
        return AgentActionResponse(**initial_info)
    
    return AgentActionResponse(**episode_history_live[-1])

# This block is for local development and running the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)