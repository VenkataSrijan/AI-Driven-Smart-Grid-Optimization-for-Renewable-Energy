# Smart Grid AI Agent

A reinforcement learning-based smart grid energy management system that uses Deep Q-Network (DQN) to optimize battery charging/discharging decisions in a renewable energy grid environment.

## 🔋 Overview

This project implements an AI agent that learns to manage energy storage in a smart grid system with:
- **Solar and Wind Generation**: Simulated renewable energy sources
- **Battery Storage**: Controllable energy storage system with realistic efficiency
- **Dynamic Demand**: Variable energy consumption patterns
- **Grid Interaction**: Import/export capabilities with penalty system

The DQN agent learns to minimize energy waste and unmet demand by making optimal battery management decisions.

## 🏗️ Architecture

### Core Components

1. **GridEnv** (`grid_env.py`): Custom OpenAI Gym-style environment
2. **DQNAgent** (`dqn_agent.py`): Deep Q-Network implementation with experience replay
3. **DataGenerator** (`data_generator.py`): Synthetic energy data generation
4. **Training Script** (`main.py`): Model training pipeline
5. **Evaluation Script** (`evaluate_agent.py`): Performance testing and visualization
6. **API Server** (`api.py`): FastAPI-based REST API for real-time control

### Key Features

- **4-dimensional observation space**: Battery charge, net energy, future energy forecast, time step
- **3-action space**: Discharge (-1), Idle (0), Charge (+1)
- **Reward system**: Heavy penalty for unmet demand, moderate penalty for waste
- **Experience replay**: Efficient learning from past experiences
- **Target network**: Stable Q-learning with periodic updates

## 📋 Requirements

### Python Dependencies
```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 4GB+ RAM

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd smart-grid-ai-agent

# Install dependencies
pip install torch numpy matplotlib fastapi uvicorn pydantic
```

### 2. Train the Agent
```bash
python main.py
```
This will:
- Train the DQN agent for 1000 episodes
- Save the model as `dqn_grid_agent_trained.pth`
- Display training progress and final performance plots

### 3. Evaluate Performance
```bash
python evaluate_agent.py
```
This will:
- Load the trained model
- Run 100 evaluation episodes
- Generate detailed performance visualizations
- Save plots as `episode_detailed_performance.png`

### 4. Start API Server
```bash
python api.py
```
The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`

## 📊 Configuration

### Environment Parameters
```python
BATTERY_CAPACITY = 500          # kWh
BATTERY_EFFICIENCY = 0.9        # 90% efficiency
FUTURE_DEMAND_LOOK_AHEAD = 3    # Hours of future data
NUM_TIME_STEPS = 48            # Steps per episode (48 hours)
```

### DQN Hyperparameters
```python
DQN_LEARNING_RATE = 0.0005
DQN_DISCOUNT_FACTOR = 0.99
DQN_EXPLORATION_START = 1.0
DQN_EXPLORATION_MIN = 0.01
DQN_EXPLORATION_DECAY = 0.998
DQN_REPLAY_BUFFER_SIZE = 50000
DQN_BATCH_SIZE = 64
```

## 🌐 API Endpoints

### Core Endpoints

- **POST `/api/predict_action`**: Get optimal action for given state
- **POST `/api/simulate_episode`**: Run complete 48-step simulation
- **GET `/api/reset_env`**: Reset environment for new episode
- **GET `/api/take_action`**: Execute single action step
- **GET `/api/current_status`**: Get current environment state
- **GET `/api/episode_history`**: Retrieve episode history data

### Example API Usage

```python
import requests

# Predict action for given state
response = requests.post("http://localhost:8000/api/predict_action", json={
    "battery_charge": 250.0,
    "net_energy": -30.0,
    "avg_future_net_energy": 15.0,
    "time_step": 12
})

action_data = response.json()
print(f"Recommended action: {action_data['action']}")
print(f"Action description: {action_data['action_description']}")
```

## 📈 Performance Metrics

The system tracks several key performance indicators:

- **Total Reward**: Composite score based on energy management efficiency
- **Unmet Demand**: Energy shortfall when demand exceeds supply and storage
- **Wasted Energy**: Excess energy that cannot be stored or used
- **Grid Import/Export**: Energy transactions with the main grid
- **Battery Utilization**: Efficiency of battery charge/discharge cycles

## 🧠 Model Architecture

### Deep Q-Network Structure
```
Input Layer (4 neurons) → FC1 (256) → FC2 (128) → FC3 (64) → Output (3 actions)
```

### Training Features
- **Experience Replay**: 50,000 transition buffer
- **Target Network**: Updated every 10 episodes
- **Gradient Clipping**: Prevents exploding gradients
- **Epsilon-Greedy Exploration**: Decaying from 1.0 to 0.01
- **Smooth L1 Loss**: Robust loss function for Q-value learning

## 📊 Data Generation

The system generates realistic energy profiles:

### Solar Generation
- Sine wave pattern for day/night cycle
- Peak generation around midday
- Random variations for weather effects
- Zero generation during night hours

### Wind Generation
- Combined sine wave and noise patterns
- Often stronger during night hours
- More variable than solar generation
- Includes random weather fluctuations

### Energy Demand
- Dual-peak pattern (morning and evening)
- Realistic household/commercial consumption
- Minimum baseline demand maintained
- Random variations throughout the day

## 🔧 Customization

### Extending the Environment
To modify the grid environment:
1. Update observation space in `grid_env.py`
2. Adjust reward function in the `step()` method
3. Modify action space as needed
4. Update scaling factors in `_scale()` method

### Training Customization
Modify hyperparameters in `main.py`:
- Increase `NUM_EPISODES` for longer training
- Adjust `DQN_LEARNING_RATE` for faster/slower learning
- Change `TARGET_UPDATE_FREQUENCY` for stability
- Modify network architecture in `dqn_agent.py`

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Not Found**: Run `main.py` to train model first
3. **API Server Issues**: Check port 8000 availability
4. **Training Instability**: Reduce learning rate or increase target update frequency

### Performance Optimization
- Use GPU for training (significant speedup)
- Increase replay buffer size for better sample efficiency
- Tune exploration parameters for your specific use case
- Monitor memory usage during training

## 📚 References

- [Deep Q-Network (DQN) Paper](https://arxiv.org/abs/1312.5602)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [PyTorch Deep Learning Framework](https://pytorch.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🎯 Future Enhancements

- [ ] Multi-agent coordination for grid-scale deployment
- [ ] Integration with real weather data APIs
- [ ] Advanced forecasting models for renewable generation
- [ ] Economic optimization with dynamic pricing
- [ ] Reinforcement learning from human feedback (RLHF)
- [ ] Real-time hardware integration capabilities

---

**Note**: This is a simulation environment for research and development purposes. For production deployment, additional safety measures, validation, and testing would be required.
