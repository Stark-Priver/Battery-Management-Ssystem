import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_battery_test_data(num_samples=5000, cycles=50):
    """
    Generate realistic battery test data with SOC and SOH for LSTM model testing
    Parameters:
        num_samples: Total number of data points (default 5000)
        cycles: Number of charge-discharge cycles (default 50)
    Returns:
        pandas.DataFrame with battery test data
    """
    np.random.seed(42)  # For reproducible results
    
    # Time stamps (1 sample per minute)
    time = [datetime.now() - timedelta(minutes=i) for i in range(num_samples)][::-1]
    
    # Simulate battery aging over cycles
    cycle_points = np.linspace(0, cycles, num_samples)
    
    # State of Health (linear degradation with some noise)
    soh = 100 - 0.04 * cycle_points + np.random.normal(0, 0.1, num_samples)
    
    # State of Charge (charge/discharge cycles with aging effects)
    soc = np.zeros(num_samples)
    soc[0] = 100  # Start fully charged
    
    # Create realistic charge/discharge pattern
    samples_per_cycle = num_samples // cycles
    for i in range(1, num_samples):
        # Current cycle phase (0-1 within each cycle)
        phase = (i % samples_per_cycle) / samples_per_cycle
        
        # Discharge faster when battery is older
        age_factor = 1 + (100 - soh[i]) / 200
        
        if phase < 0.8:  # Discharging phase
            soc[i] = soc[i-1] - (0.15 * age_factor) + np.random.normal(0, 0.02)
        else:  # Charging phase
            soc[i] = soc[i-1] + (0.25 * age_factor) + np.random.normal(0, 0.03)
            
        # Ensure SOC stays between 0-100%
        soc[i] = np.clip(soc[i], 0, 100)
    
    # Voltage depends on SOC and SOH with some noise
    voltage = (3.2 + 0.8 * (soc/100)) * (soh/100) + np.random.normal(0, 0.02, num_samples)
    
    # Current pattern - higher during discharge, lower during charge
    current = np.where(soc < soc[np.maximum(0, i-10):i].mean(), 
                       np.random.uniform(3.5, 5.0),  # Discharging
                       np.random.uniform(0.5, 2.0))  # Charging
    
    # Temperature depends on current and SOH
    temperature = (25 + 0.5 * current * (1 + (100-soh)/100) + 
                  5 * np.sin(2*np.pi*cycle_points/cycles) + 
                  np.random.normal(0, 0.5, num_samples))
    
    # Internal resistance increases with SOH degradation
    internal_resistance = (0.1 + (100-soh)/250 + 
                          np.random.normal(0, 0.002, num_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time,
        'cycle_number': np.floor(cycle_points).astype(int),
        'voltage': np.clip(voltage, 2.8, 4.2),
        'current': np.clip(current, 0.1, 5.5),
        'temperature': np.clip(temperature, 20, 45),
        'internal_resistance': np.clip(internal_resistance, 0.1, 0.3),
        'soc_actual': np.clip(soc, 0, 100),
        'soh_actual': np.clip(soh, 70, 100)  # SOH never goes below 70%
    })
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate test data
    print("Generating battery test data...")
    test_data = generate_battery_test_data()
    
    # Save to CSV
    filepath = os.path.join('data', 'battery_test_data.csv')
    test_data.to_csv(filepath, index=False)
    
    print(f"Successfully generated {len(test_data)} samples")
    print(f"Data saved to: {filepath}")
    print("\nSample of generated data:")
    print(test_data.head())