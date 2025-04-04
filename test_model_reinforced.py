import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import os
import random
from collections import deque
from datetime import datetime

class EnergyFlowOptimizer:
    def __init__(self):
        self.state_size = 6  # [SOC, SOH, PV_power, Load_demand, Time_of_day, Battery_temp]
        self.action_size = 3  # [0: Max self-consumption, 1: Max battery charging, 2: Balanced]
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class BatteryEstimatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced BMS with Energy Flow Optimization")
        self.root.geometry("1400x1200")
        
        # Initialize variables
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.df = None
        self.sequence_length = 10
        self.current_sequence = []
        self.soc_history = []
        self.soh_history = []
        self.time_points = []
        self.autoplay_running = False
        
        # Energy flow variables
        self.energy_optimizer = EnergyFlowOptimizer()
        self.batch_size = 32
        self.pv_power = 0
        self.load_demand = 0
        self.energy_flows = []
        self.pv_history = []
        self.load_history = []
        self.current_action = 0
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Create GUI
        self.create_widgets()
        
        # Load model and scalers
        self.load_model_and_scalers()
    
    def load_model_and_scalers(self):
        try:
            self.model = load_model('models/battery_lstm.keras')
            self.x_scaler = joblib.load('models/x_scaler.save')
            self.y_scaler = joblib.load('models/y_scaler.save')
            self.model_status_label.config(text="Model loaded successfully", foreground="green")
            self.predict_button.config(state=tk.NORMAL)
            self.autoplay_button.config(state=tk.NORMAL)
        except Exception as e:
            self.model_status_label.config(text=f"Error loading model: {str(e)}", foreground="red")
            self.predict_button.config(state=tk.DISABLED)
            self.autoplay_button.config(state=tk.DISABLED)
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model status frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_status_label = ttk.Label(model_frame, text="Loading model...")
        self.model_status_label.pack()
        
        # File upload frame
        file_frame = ttk.LabelFrame(main_frame, text="Data File", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No data file loaded")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.load_button = ttk.Button(file_frame, text="Load CSV", command=self.load_file)
        self.load_button.pack(side=tk.RIGHT, padx=5)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Battery Parameters", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Battery parameter inputs
        ttk.Label(input_frame, text="Voltage (V):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.voltage_entry = ttk.Entry(input_frame)
        self.voltage_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Current (A):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.current_entry = ttk.Entry(input_frame)
        self.current_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Temperature (°C):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.temp_entry = ttk.Entry(input_frame)
        self.temp_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Internal Resistance (Ω):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.resistance_entry = ttk.Entry(input_frame)
        self.resistance_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Energy flow frame
        energy_frame = ttk.LabelFrame(main_frame, text="Energy Flow Parameters", padding="10")
        energy_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(energy_frame, text="PV Power (W):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.pv_entry = ttk.Entry(energy_frame)
        self.pv_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(energy_frame, text="Load Demand (W):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.load_entry = ttk.Entry(energy_frame)
        self.load_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict SOC/SOH", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.optimize_button = ttk.Button(button_frame, text="Optimize Energy Flow", command=self.optimize_energy_flow)
        self.optimize_button.pack(side=tk.LEFT, padx=5)
        
        self.autoplay_button = ttk.Button(button_frame, text="Autoplay", command=self.autoplay_data, state=tk.DISABLED)
        self.autoplay_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_data)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_autoplay)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        # SOC result
        soc_frame = ttk.Frame(result_frame)
        soc_frame.pack(side=tk.LEFT, expand=True)
        ttk.Label(soc_frame, text="State of Charge (SOC):", font=('Arial', 10, 'bold')).pack()
        self.soc_label = ttk.Label(soc_frame, text="0.00%", font=('Arial', 12))
        self.soc_label.pack()
        
        # SOH result
        soh_frame = ttk.Frame(result_frame)
        soh_frame.pack(side=tk.RIGHT, expand=True)
        ttk.Label(soh_frame, text="State of Health (SOH):", font=('Arial', 10, 'bold')).pack()
        self.soh_label = ttk.Label(soh_frame, text="0.00%", font=('Arial', 12))
        self.soh_label.pack()
        
        # Visualization frame
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figures
        self.fig, (self.ax_soc, self.ax_soh) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Configure SOC plot
        self.ax_soc.set_title("State of Charge (SOC)")
        self.ax_soc.set_xlabel("Time Step")
        self.ax_soc.set_ylabel("SOC (%)")
        self.ax_soc.grid(True)
        self.ax_soc.set_ylim(0, 100)
        self.line_soc_pred, = self.ax_soc.plot([], [], 'b-', label='Estimated SOC')
        self.line_soc_actual, = self.ax_soc.plot([], [], 'r--', label='Actual SOC')
        self.ax_soc.legend()
        
        # Configure SOH plot
        self.ax_soh.set_title("State of Health (SOH)")
        self.ax_soh.set_xlabel("Time Step")
        self.ax_soh.set_ylabel("SOH (%)")
        self.ax_soh.grid(True)
        self.ax_soh.set_ylim(0, 100)
        self.line_soh_pred, = self.ax_soh.plot([], [], 'g-', label='Estimated SOH')
        self.line_soh_actual, = self.ax_soh.plot([], [], 'm--', label='Actual SOH')
        self.ax_soh.legend()
        
        # Energy flow visualization
        self.energy_fig, self.energy_ax = plt.subplots(figsize=(12, 3))
        self.energy_ax.set_title("Energy Flow Optimization")
        self.energy_ax.set_xlabel("Time Step")
        self.energy_ax.set_ylabel("Power (W)")
        self.energy_ax.grid(True)
        self.line_pv_load, = self.energy_ax.plot([], [], 'y-', label='PV to Load')
        self.line_pv_batt, = self.energy_ax.plot([], [], 'c-', label='PV to Battery')
        self.line_batt_load, = self.energy_ax.plot([], [], 'm-', label='Battery to Load')
        self.energy_ax.legend()
        
        # Power vs Demand visualization
        self.power_fig, self.power_ax = plt.subplots(figsize=(12, 3))
        self.power_ax.set_title("PV Power and Load Demand")
        self.power_ax.set_xlabel("Time Step")
        self.power_ax.set_ylabel("Power (W)")
        self.power_ax.grid(True)
        self.line_pv, = self.power_ax.plot([], [], 'g-', label='PV Power')
        self.line_load, = self.power_ax.plot([], [], 'r-', label='Load Demand')
        self.power_ax.legend()
        
        # Create canvases
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=vis_frame)
        self.energy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.power_canvas = FigureCanvasTkAgg(self.power_fig, master=vis_frame)
        self.power_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Adjust layout
        self.fig.tight_layout()
        self.energy_fig.tight_layout()
        self.power_fig.tight_layout()
    
    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            initialdir=os.path.join(os.getcwd(), 'data')
        )
        if filepath:
            try:
                self.df = pd.read_csv(filepath)
                
                required_cols = ['voltage', 'current', 'temperature', 'internal_resistance']
                if not all(col in self.df.columns for col in required_cols):
                    raise ValueError("CSV missing required columns")
                
                self.file_label.config(text=f"Loaded: {os.path.basename(filepath)}")
                self.reset_data()
                
                if len(self.df) > 0:
                    first_row = self.df.iloc[0]
                    self.voltage_entry.delete(0, tk.END)
                    self.voltage_entry.insert(0, f"{first_row['voltage']:.2f}")
                    self.current_entry.delete(0, tk.END)
                    self.current_entry.insert(0, f"{first_row['current']:.2f}")
                    self.temp_entry.delete(0, tk.END)
                    self.temp_entry.insert(0, f"{first_row['temperature']:.2f}")
                    self.resistance_entry.delete(0, tk.END)
                    self.resistance_entry.insert(0, f"{first_row['internal_resistance']:.4f}")
                
            except Exception as e:
                self.file_label.config(text=f"Error: {str(e)}")
                self.df = None
    
    def predict(self):
        if self.model is None:
            self.soc_label.config(text="Model not loaded", foreground="red")
            self.soh_label.config(text="Model not loaded", foreground="red")
            return
            
        try:
            voltage = float(self.voltage_entry.get())
            current = float(self.current_entry.get())
            temp = float(self.temp_entry.get())
            resistance = float(self.resistance_entry.get())
            
            new_data = np.array([[voltage, current, temp, resistance]])
            scaled_data = self.x_scaler.transform(new_data)
            
            if len(self.current_sequence) >= self.sequence_length:
                self.current_sequence.pop(0)
            self.current_sequence.append(scaled_data[0])
            
            if len(self.current_sequence) == self.sequence_length:
                X_input = np.array([self.current_sequence])
                
                soc_pred, soh_pred = self.model.predict(X_input, verbose=0)
                
                preds = self.y_scaler.inverse_transform(
                    np.concatenate([soc_pred, soh_pred], axis=1)
                )
                soc_value = preds[0, 0]
                soh_value = preds[0, 1]
                
                self.soc_history.append(soc_value)
                self.soh_history.append(soh_value)
                if len(self.time_points) == 0:
                    self.time_points.append(0)
                else:
                    self.time_points.append(self.time_points[-1] + 1)
                
                self.soc_label.config(text=f"{soc_value:.2f}%")
                self.soh_label.config(text=f"{soh_value:.2f}%")
                self.update_plots()
            else:
                self.soc_label.config(text=f"Collecting data ({len(self.current_sequence)}/{self.sequence_length})")
                self.soh_label.config(text="Waiting...")
                
        except ValueError as e:
            self.soc_label.config(text="Invalid input", foreground="red")
            self.soh_label.config(text="Invalid input", foreground="red")
        except Exception as e:
            self.soc_label.config(text=f"Error: {str(e)}", foreground="red")
            self.soh_label.config(text=f"Error: {str(e)}", foreground="red")
    
    def optimize_energy_flow(self):
        try:
            # Get current system state
            self.pv_power = float(self.pv_entry.get())
            self.load_demand = float(self.load_entry.get())
            time_of_day = datetime.now().hour / 24  # Normalized [0,1]
            
            # Store PV and load values
            self.pv_history.append(self.pv_power)
            self.load_history.append(self.load_demand)
            
            # Get battery state
            soc = float(self.soc_label.cget("text").replace("%", "")) / 100
            soh = float(self.soh_label.cget("text").replace("%", "")) / 100
            temp = float(self.temp_entry.get())
            
            # Create state vector
            state = np.array([[soc, soh, self.pv_power, self.load_demand, 
                             time_of_day, temp]])
            
            # Get action from DRL agent
            self.current_action = self.energy_optimizer.act(state)
            
            # Execute action (simulated)
            reward, next_state, done = self.simulate_energy_flow(self.current_action, state)
            
            # Remember the experience
            self.energy_optimizer.remember(state, self.current_action, reward, next_state, done)
            
            # Train the model
            if len(self.energy_optimizer.memory) > self.batch_size:
                self.energy_optimizer.replay(self.batch_size)
            
            # Update visualizations
            self.update_energy_flow_visualization()
            self.update_power_demand_plot()
            self.update_plots()
            
        except Exception as e:
            print(f"Optimization error: {str(e)}")
    
    def simulate_energy_flow(self, action, state):
        soc, soh, pv_power, load_demand, time_of_day, temp = state[0]
        battery_capacity = 5000  # 5kWh battery
        
        if action == 0:  # Max self-consumption
            pv_to_load = min(pv_power, load_demand)
            remaining_pv = pv_power - pv_to_load
            pv_to_battery = min(remaining_pv, (1-soc) * battery_capacity)
            battery_to_load = max(0, load_demand - pv_to_load)
            
        elif action == 1:  # Max battery charging
            pv_to_battery = min(pv_power, (1-soc) * battery_capacity)
            remaining_pv = pv_power - pv_to_battery
            pv_to_load = min(remaining_pv, load_demand)
            battery_to_load = max(0, load_demand - pv_to_load)
            
        else:  # Balanced approach
            pv_to_load = min(pv_power * 0.7, load_demand)
            pv_to_battery = min(pv_power * 0.3, (1-soc) * battery_capacity)
            battery_to_load = max(0, load_demand - pv_to_load)
        
        # Calculate reward
        reward = 0
        reward += pv_to_load * 0.2       # Reward for direct PV usage
        reward += pv_to_battery * 0.1    # Reward for storing energy
        reward -= battery_to_load * 0.05 # Penalty for battery discharge
        reward -= (1-soh) * 10           # Penalty for battery degradation
        
        # Calculate next state
        new_soc = soc + (pv_to_battery/battery_capacity) - (battery_to_load/battery_capacity)
        new_soh = soh - (battery_to_load/(battery_capacity*10))  # Simplified degradation
        
        next_state = np.array([[new_soc, new_soh, pv_power, load_demand, 
                              (time_of_day + 0.01) % 1, temp]])
        
        # Store energy flows
        self.energy_flows.append({
            'time': len(self.energy_flows),
            'pv_to_load': pv_to_load,
            'pv_to_battery': pv_to_battery,
            'battery_to_load': battery_to_load,
            'action': action
        })
        
        done = len(self.energy_flows) >= 1000  # Example termination condition
        
        return reward, next_state, done
    
    def update_energy_flow_visualization(self):
        if len(self.energy_flows) == 0:
            return
            
        # Show last 50 steps
        start_idx = max(0, len(self.energy_flows) - 50)
        times = [x['time'] for x in self.energy_flows[start_idx:]]
        pv_load = [x['pv_to_load'] for x in self.energy_flows[start_idx:]]
        pv_batt = [x['pv_to_battery'] for x in self.energy_flows[start_idx:]]
        batt_load = [x['battery_to_load'] for x in self.energy_flows[start_idx:]]
        
        self.line_pv_load.set_data(times, pv_load)
        self.line_pv_batt.set_data(times, pv_batt)
        self.line_batt_load.set_data(times, batt_load)
        
        self.energy_ax.set_xlim(min(times), max(times))
        max_power = max(max(pv_load), max(pv_batt), max(batt_load)) * 1.1
        self.energy_ax.set_ylim(0, max(100, max_power))
        
        action_names = ["Max Self-Consumption", "Max Charge", "Balanced"]
        current_action = self.energy_flows[-1]['action']
        self.energy_ax.set_title(f"Energy Flow (Current Strategy: {action_names[current_action]})")
        
        self.energy_canvas.draw()
    
    def update_power_demand_plot(self):
        if len(self.pv_history) == 0 or len(self.load_history) == 0:
            return
            
        # Show last 50 steps
        start_idx = max(0, len(self.pv_history) - 50)
        times = list(range(start_idx, len(self.pv_history)))
        pv_values = self.pv_history[start_idx:]
        load_values = self.load_history[start_idx:]
        
        self.line_pv.set_data(times, pv_values)
        self.line_load.set_data(times, load_values)
        
        self.power_ax.set_xlim(min(times), max(times))
        max_power = max(max(pv_values), max(load_values)) * 1.1
        self.power_ax.set_ylim(0, max(100, max_power))
        
        self.power_canvas.draw()
    
    def update_plots(self):
        if len(self.time_points) > 0:
            # Update SOC plot
            if len(self.soc_history) > 0:
                self.line_soc_pred.set_data(self.time_points[:len(self.soc_history)], self.soc_history)
                self.ax_soc.set_xlim(0, max(self.time_points))
            
            # Update SOH plot
            if len(self.soh_history) > 0:
                self.line_soh_pred.set_data(self.time_points[:len(self.soh_history)], self.soh_history)
                self.ax_soh.set_xlim(0, max(self.time_points))
            
            self.canvas.draw()
        
        self.update_energy_flow_visualization()
        self.update_power_demand_plot()
    
    def autoplay_data(self):
        if self.df is None or len(self.df) == 0:
            self.soc_label.config(text="Load data first", foreground="red")
            return
            
        if self.autoplay_running:
            return
            
        self.autoplay_running = True
        self.reset_data()
        
        def process_next_row(i):
            if not self.autoplay_running or i >= len(self.df):
                self.autoplay_running = False
                return
                
            row = self.df.iloc[i]
            
            # Update battery parameters
            self.voltage_entry.delete(0, tk.END)
            self.voltage_entry.insert(0, f"{row['voltage']:.2f}")
            self.current_entry.delete(0, tk.END)
            self.current_entry.insert(0, f"{row['current']:.2f}")
            self.temp_entry.delete(0, tk.END)
            self.temp_entry.insert(0, f"{row['temperature']:.2f}")
            self.resistance_entry.delete(0, tk.END)
            self.resistance_entry.insert(0, f"{row['internal_resistance']:.4f}")
            
            # Make SOC/SOH prediction
            self.predict()
            
            # Update plots with actual values if available
            if 'soc_actual' in self.df.columns:
                self.line_soc_actual.set_data(np.arange(i+1), self.df['soc_actual'].values[:i+1])
            if 'soh_actual' in self.df.columns:
                self.line_soh_actual.set_data(np.arange(i+1), self.df['soh_actual'].values[:i+1])
            
            # Simulate random PV and load for energy flow optimization
            if i % 5 == 0:  # Optimize every 5 steps
                self.pv_entry.delete(0, tk.END)
                self.pv_entry.insert(0, f"{random.uniform(100, 2000):.1f}")
                self.load_entry.delete(0, tk.END)
                self.load_entry.insert(0, f"{random.uniform(50, 1500):.1f}")
                self.optimize_energy_flow()
            
            self.update_plots()
            self.root.after(200, process_next_row, i+1)
        
        process_next_row(0)
    
    def stop_autoplay(self):
        self.autoplay_running = False
    
    def reset_data(self):
        self.current_sequence = []
        self.soc_history = []
        self.soh_history = []
        self.time_points = []
        self.energy_flows = []
        self.pv_history = []
        self.load_history = []
        
        self.soc_label.config(text="0.00%", foreground="black")
        self.soh_label.config(text="0.00%", foreground="black")
        
        self.line_soc_pred.set_data([], [])
        self.line_soc_actual.set_data([], [])
        self.line_soh_pred.set_data([], [])
        self.line_soh_actual.set_data([], [])
        
        self.line_pv_load.set_data([], [])
        self.line_pv_batt.set_data([], [])
        self.line_batt_load.set_data([], [])
        
        self.line_pv.set_data([], [])
        self.line_load.set_data([], [])
        
        self.update_plots()

if __name__ == "__main__":
    root = tk.Tk()
    app = BatteryEstimatorApp(root)
    root.mainloop()