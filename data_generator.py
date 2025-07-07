# data_generator.py
import numpy as np
import matplotlib.pyplot as plt # Make sure to re-add this import if you want plotting later

class DataGenerator: # <--- ADDED: Define the class
    def __init__(self, num_time_steps): # <--- ADDED: Constructor
        self.num_time_steps = num_time_steps
        self.hours_in_day = 24
        self.days = num_time_steps // self.hours_in_day

    def generate_data(self): # <--- RENAMED: Changed from generate_daily_profiles to generate_data
        """
        Generates simulated daily profiles for solar, wind, and demand.
        This is simplified for now, assuming 24 hourly time steps per day.
        """
        all_solar_gen = []
        all_wind_gen = []
        all_demand = []

        for day in range(self.days): # <--- MODIFIED: Use self.days
            # Solar generation: Peaks around midday (hour 12-14), zero at night
            # Uses a sine wave to simulate daily pattern
            hours = np.linspace(0, 24, self.hours_in_day) # <--- MODIFIED: Use self.hours_in_day
            solar_profile = np.sin(hours * np.pi / 12 - np.pi / 2) # Sine wave for day/night
            solar_profile = np.maximum(0, solar_profile) * 50 + np.random.rand(self.hours_in_day) * 5 # <--- MODIFIED: Use self.hours_in_day
            # Scale solar to be more realistic, e.g., max 50 kWh per hour for a small setup

            # Wind generation: More random, but often stronger at night or certain times
            # Using a combination of sine and noise for variability
            wind_profile = (np.sin(hours * np.pi / 8 + np.pi / 4) * 20 + 30) + np.random.rand(self.hours_in_day) * 10 # <--- MODIFIED
            wind_profile = np.maximum(0, wind_profile) # Ensure no negative generation
            # Scale wind, e.g., max 60 kWh per hour

            # Demand: Peaks in morning (e.g., hour 8) and evening (e.g., hour 19)
            demand_profile = (
                (np.sin(hours * np.pi / 12 - np.pi / 4) * 30 + 40) + # Base daily pattern
                (np.sin(hours * np.pi / 6 + np.pi) * 15 + 10) +      # Secondary peak/trough
                np.random.rand(self.hours_in_day) * 8               # Add some noise
            )
            demand_profile = np.maximum(10, demand_profile) # Ensure minimum demand, e.g., min 10 kWh

            all_solar_gen.extend(solar_profile)
            all_wind_gen.extend(wind_profile)
            all_demand.extend(demand_profile)

        return np.array(all_solar_gen), np.array(all_wind_gen), np.array(all_demand)

    def plot_daily_profiles(self, solar_data, wind_data, demand_data): # <--- ADDED: Re-add the plotting method
        """
        Plots the generated solar, wind, and demand profiles for the entire duration.
        """
        time_hours = np.arange(self.num_time_steps)

        plt.figure(figsize=(15, 7))
        plt.plot(time_hours, solar_data, label='Solar Generation (kWh/h)', color='orange')
        plt.plot(time_hours, wind_data, label='Wind Generation (kWh/h)', color='skyblue')
        plt.plot(time_hours, demand_data, label='Demand (kWh/h)', color='red', linestyle='--')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Energy (kWh/h)')
        plt.title(f'Simulated Daily Energy Profiles Over {self.days} Days')
        plt.xticks(np.arange(0, self.num_time_steps + 1, self.hours_in_day),
                   [f'Day {i//self.hours_in_day + 1} Hr 0' for i in np.arange(0, self.num_time_steps + 1, self.hours_in_day)])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example usage:
    data_gen = DataGenerator(num_time_steps=48)
    solar, wind, demand = data_gen.generate_data()
    data_gen.plot_daily_profiles(solar, wind, demand)