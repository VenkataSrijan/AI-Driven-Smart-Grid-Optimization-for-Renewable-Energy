import numpy as np
import random
from data_generator import DataGenerator

class GridEnv:
    def __init__(self, battery_capacity, battery_efficiency, look_ahead, num_steps, initial_solar_data=None, initial_wind_data=None, initial_demand_data=None):
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.future_demand_look_ahead = look_ahead
        self.num_time_steps = num_steps
        self.action_space = [-1, 0, 1] # -1: discharge, 0: idle, 1: charge
        self.observation_space_size = 4 # Confirmed number of features returned by _scale
        self.data_gen = DataGenerator(num_steps) # Instance of data generator

        # Allows pre-loading specific data for consistent testing
        self._solar = initial_solar_data
        self._wind = initial_wind_data
        self._demand = initial_demand_data

    def reset(self):
        """Resets the environment to its initial state for a new episode."""
        self.current_battery_charge = self.battery_capacity / 2 # Start with half charge
        
        # Load or generate new data for the episode
        if self._solar is not None:
            self.solar = self._solar
            self.wind = self._wind
            self.demand = self._demand
        else:
            self.solar, self.wind, self.demand = self.data_gen.generate_data()
            
        self.current_time_step = 0 # Reset time step
        self._update_future() # Pre-calculate future data for observation

        # Calculate initial net energy and average future net energy
        net = self.solar[0] + self.wind[0] - self.demand[0]
        # Average net energy over the look-ahead window
        avg_net = np.mean([(self.future_solar[i] + self.future_wind[i]) - self.future_demand[i] for i in range(self.future_demand_look_ahead)])
        
        return self._scale(self.current_battery_charge, net, avg_net, self.current_time_step)

    def step(self, action_idx):
        """Takes an action in the environment and returns the next state, reward, done flag, and info."""
        action = self.action_space[action_idx] # Convert action index to value (-1, 0, or 1)
        t = self.current_time_step
        
        # Current energy generation and demand
        solar = self.solar[t]
        wind = self.wind[t]
        demand = self.demand[t]
        
        net = solar + wind - demand # Net energy: positive for surplus, negative for deficit
        unmet = waste = g_import = g_export = 0 # Initialize metrics for the step

        # Apply battery action
        if net > 0 and action == 1: # Surplus energy, charge battery
            capacity_to_fill = self.battery_capacity - self.current_battery_charge
            charge_amount = min(net, capacity_to_fill / self.battery_efficiency)
            self.current_battery_charge += charge_amount * self.battery_efficiency
            net -= charge_amount # Remaining net energy after charging
        elif net < 0 and action == -1: # Deficit energy, discharge battery
            deficit_to_cover = -net
            discharge_amount = min(deficit_to_cover, self.current_battery_charge * self.battery_efficiency)
            self.current_battery_charge -= discharge_amount / self.battery_efficiency
            net += discharge_amount # Remaining net energy after discharging

        # Calculate grid interaction (import/export) and penalties (unmet/waste)
        if net > 0: # Energy surplus after battery action
            waste = g_export = net
        elif net < 0: # Energy deficit after battery action
            unmet = g_import = -net

        # Clip battery charge to stay within capacity limits
        self.current_battery_charge = np.clip(self.current_battery_charge, 0, self.battery_capacity)
        
        self.current_time_step += 1 # Advance time step
        done = self.current_time_step >= self.num_time_steps # Check if episode is finished
        
        # --- Reward Function (from working solution) ---
        # Penalizes unmet demand heavily, and wasted energy moderately
        reward = -(unmet * 50) - (waste * 0.5)

        next_obs = None
        if not done:
            self._update_future() # Update future data for the next observation
            next_net = self.solar[t + 1] + self.wind[t + 1] - self.demand[t + 1]
            next_avg = np.mean([(self.future_solar[i] + self.future_wind[i]) - self.future_demand[i] for i in range(self.future_demand_look_ahead)])
            next_obs = self._scale(self.current_battery_charge, next_net, next_avg, self.current_time_step)

        info = {
            "current_solar": solar,
            "current_wind": wind,
            "current_demand": demand,
            "grid_import": g_import,
            "grid_export": g_export,
            "unmet_demand": unmet,
            "wasted_energy": waste
        }
        
        return next_obs, reward, done, info

    def _update_future(self):
        """Updates the future solar, wind, and demand data based on the current time step."""
        # Define the slice for future data
        i = self.current_time_step + 1
        j = i + self.future_demand_look_ahead
        
        # Get future data, converting to list for easier manipulation
        self.future_solar = self.solar[i:j].tolist()
        self.future_wind = self.wind[i:j].tolist()
        self.future_demand = self.demand[i:j].tolist()
        
        # Pad with zeros if future_demand_look_ahead extends beyond available data
        while len(self.future_solar) < self.future_demand_look_ahead:
            self.future_solar.append(0.0)
            self.future_wind.append(0.0)
            self.future_demand.append(0.0)

    def _scale(self, charge, net, future_net, time_step):
        """Scales observation components to be within a reasonable range (0 to 1)."""
        return np.array([
            charge / self.battery_capacity, # Battery charge as a fraction of capacity
            (net + 200) / 400, # Net energy normalized (assuming range -200 to 200)
            (future_net + 200) / 400, # Average future net energy normalized
            time_step / (self.num_time_steps - 1) # Current time step normalized
        ], dtype=np.float32)