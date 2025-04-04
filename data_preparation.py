import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_battery_data(num_samples=1000):
    """Generate synthetic battery data with SOC and SOH"""
    np.random.seed(42)
    
    # Base parameters
    time = [datetime.now() - timedelta(minutes=i) for i in range(num_samples)][::-1]
    
    # Simulate battery aging over time
    soh = np.linspace(100, 80, num_samples) + np.random.normal(0, 0.5, num_samples)
    
    # SOC depends on both usage and SOH
    soc = np.zeros(num_samples)
    soc[0] = 100
    for i in range(1, num_samples):
        soc[i] = soc[i-1] - (0.1 + (100-soh[i])/500) + np.random.normal(0, 0.2)
        soc[i] = np.clip(soc[i], 0, 100)
    
    # Voltage depends on SOC and SOH
    voltage = 3.0 + 1.2 * (soc/100) * (soh/100) + np.random.normal(0, 0.05, num_samples)
    
    # Current and temperature patterns
    current = np.random.uniform(0.1, 5.0, num_samples)
    temperature = 25 + 10 * np.sin(np.linspace(0, 10*np.pi, num_samples)) + np.random.normal(0, 2, num_samples)
    
    # Internal resistance increases as SOH decreases
    internal_resistance = 0.1 + (100-soh)/200 + np.random.normal(0, 0.01, num_samples)
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': time,
        'voltage': np.clip(voltage, 2.7, 4.2),
        'current': np.clip(current, 0, 5.5),
        'temperature': np.clip(temperature, 15, 45),
        'internal_resistance': np.clip(internal_resistance, 0.1, 0.3),
        'soc_actual': np.clip(soc, 0, 100),
        'soh_actual': np.clip(soh, 0, 100)
    })
    
    return df

if __name__ == "__main__":
    # Generate and save test data
    test_data = generate_battery_data(5000)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join('data', 'battery_test_data.csv')
    test_data.to_csv(filepath, index=False)
    print(f"Test data generated and saved as '{filepath}'")