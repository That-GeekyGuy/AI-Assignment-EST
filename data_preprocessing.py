"""
Data Preprocessing Module for Turbofan Engine RUL Prediction
This module handles data loading, cleaning, feature engineering, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os

class TurbofanDataPreprocessor:
    """Class to preprocess turbofan engine degradation simulation data"""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
        self.train_rul = None
        self.test_rul = None
        
    def load_data(self, train_file='train_FD001.txt', test_file='test_FD001.txt', 
                  rul_file='RUL_FD001.txt'):
        """
        Load training and test data from text files
        
        The dataset format:
        - Column 1: Unit number
        - Column 2: Time in cycles
        - Columns 3-5: Operational settings
        - Columns 6-26: Sensor measurements (21 sensors)
        """
        if self.data_path:
            train_path = os.path.join(self.data_path, train_file)
            test_path = os.path.join(self.data_path, test_file)
            rul_path = os.path.join(self.data_path, rul_file)
        else:
            train_path = train_file
            test_path = test_file
            rul_path = rul_file
        
        # Define column names
        operational_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        columns = ['unit_id', 'time_cycles'] + operational_settings + sensor_columns
        
        # Load training data
        print("Loading training data...")
        # Use whitespace separator and skip trailing whitespace
        self.train_data = pd.read_csv(train_path, sep='\s+', header=None, engine='python')
        self.train_data = self.train_data.dropna(axis=1, how='all')  # Remove empty columns
        # Assign column names based on actual number of columns
        n_cols = self.train_data.shape[1]
        if n_cols <= len(columns):
            self.train_data.columns = columns[:n_cols]
        else:
            # If more columns than expected, use what we have
            self.train_data.columns = columns + [f'extra_{i}' for i in range(n_cols - len(columns))]
        
        # Load test data
        print("Loading test data...")
        self.test_data = pd.read_csv(test_path, sep='\s+', header=None, engine='python')
        self.test_data = self.test_data.dropna(axis=1, how='all')
        n_cols = self.test_data.shape[1]
        if n_cols <= len(columns):
            self.test_data.columns = columns[:n_cols]
        else:
            self.test_data.columns = columns + [f'extra_{i}' for i in range(n_cols - len(columns))]
        
        # Load RUL data for test set
        print("Loading RUL data...")
        self.test_rul = pd.read_csv(rul_path, sep='\s+', header=None, engine='python')
        self.test_rul = self.test_rul.dropna(axis=1, how='all')
        # Take only the first column (RUL values)
        self.test_rul = self.test_rul.iloc[:, [0]]
        self.test_rul.columns = ['RUL']
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Test RUL shape: {self.test_rul.shape}")
        
        return self.train_data, self.test_data, self.test_rul
    
    def calculate_rul(self, data):
        """
        Calculate RUL for training data (RUL decreases as time increases)
        
        Args:
            data: DataFrame with time_cycles and unit_id columns
            
        Returns:
            RUL values for each row
        """
        rul = []
        for unit_id in data['unit_id'].unique():
            unit_data = data[data['unit_id'] == unit_id]
            max_cycles = unit_data['time_cycles'].max()
            unit_rul = max_cycles - unit_data['time_cycles']
            rul.extend(unit_rul.values)
        
        return np.array(rul)
    
    def prepare_train_data(self, use_sensors_only=True):
        """
        Prepare training data with RUL calculation
        
        Args:
            use_sensors_only: If True, use only sensor data (exclude operational settings)
            
        Returns:
            X_train, y_train (RUL)
        """
        if self.train_data is None:
            raise ValueError("Training data not loaded. Call load_data() first.")
        
        # Calculate RUL for training data
        self.train_rul = self.calculate_rul(self.train_data)
        
        # Select features
        if use_sensors_only:
            # Use only sensor measurements
            feature_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        else:
            # Use operational settings and sensors
            feature_cols = [col for col in self.train_data.columns 
                          if col not in ['unit_id', 'time_cycles']]
        
        X_train = self.train_data[feature_cols].values
        y_train = self.train_rul
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Training RUL shape: {y_train.shape}")
        
        return X_train, y_train, feature_cols
    
    def prepare_test_data(self, feature_cols):
        """
        Prepare test data with actual RUL values
        
        Args:
            feature_cols: List of feature column names to use
            
        Returns:
            X_test, y_test (RUL)
        """
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_data() first.")
        
        # Get the last record for each unit in test data
        test_data_last = self.test_data.groupby('unit_id').tail(1)
        X_test = test_data_last[feature_cols].values
        y_test = self.test_rul['RUL'].values
        
        print(f"Test features shape: {X_test.shape}")
        print(f"Test RUL shape: {y_test.shape}")
        
        return X_test, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using MinMaxScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Scaled X_train, X_test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def create_sequences(self, data, rul, sequence_length=50, step_size=1):
        """
        Create sequences for LSTM/RNN input
        
        Args:
            data: Feature data (n_samples, n_features)
            rul: RUL values (n_samples,)
            sequence_length: Length of sequences
            step_size: Step size for creating sequences
            
        Returns:
            X_sequences, y_sequences
        """
        X_sequences = []
        y_sequences = []
        
        # Group by unit_id if available in original data
        if hasattr(self, 'train_data') and 'unit_id' in self.train_data.columns:
            for unit_id in self.train_data['unit_id'].unique():
                unit_mask = self.train_data['unit_id'] == unit_id
                unit_data = data[unit_mask]
                unit_rul = rul[unit_mask]
                
                for i in range(0, len(unit_data) - sequence_length + 1, step_size):
                    X_sequences.append(unit_data[i:i + sequence_length])
                    y_sequences.append(unit_rul[i + sequence_length - 1])
        else:
            # Fallback: create sequences from entire dataset
            for i in range(0, len(data) - sequence_length + 1, step_size):
                X_sequences.append(data[i:i + sequence_length])
                y_sequences.append(rul[i + sequence_length - 1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def visualize_data_distribution(self, save_path='visualizations/data_distribution.png'):
        """
        Visualize data distribution and characteristics
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        os.makedirs('visualizations', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RUL distribution
        train_rul = self.calculate_rul(self.train_data)
        axes[0, 0].hist(train_rul, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Remaining Useful Life (Cycles)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RUL Distribution in Training Data')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Number of cycles per engine
        cycles_per_engine = self.train_data.groupby('unit_id')['time_cycles'].max()
        axes[0, 1].hist(cycles_per_engine, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Maximum Cycles per Engine')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Engine Lifespan')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sample engine degradation (RUL over time)
        sample_engines = self.train_data['unit_id'].unique()[:5]
        for engine_id in sample_engines:
            engine_data = self.train_data[self.train_data['unit_id'] == engine_id]
            engine_rul = self.calculate_rul(engine_data)
            axes[1, 0].plot(engine_data['time_cycles'], engine_rul, label=f'Engine {engine_id}')
        axes[1, 0].set_xlabel('Time Cycles')
        axes[1, 0].set_ylabel('RUL (Cycles)')
        axes[1, 0].set_title('RUL Degradation Over Time (Sample Engines)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sensor correlation heatmap (sample sensors)
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        sample_sensors = sensor_cols[:10]  # First 10 sensors
        corr_matrix = self.train_data[sample_sensors].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
        axes[1, 1].set_title('Sensor Correlation Heatmap (Sample)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data distribution visualization saved to {save_path}")
        plt.close()
    
    def visualize_sensor_trends(self, n_engines=3, n_sensors=6, save_path='visualizations/sensor_trends.png'):
        """
        Visualize sensor trends over time for sample engines
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        os.makedirs('visualizations', exist_ok=True)
        
        sensor_cols = [col for col in self.train_data.columns if col.startswith('sensor_')]
        selected_sensors = sensor_cols[:n_sensors]
        sample_engines = self.train_data['unit_id'].unique()[:n_engines]
        
        fig, axes = plt.subplots(n_engines, n_sensors, figsize=(20, 12))
        if n_engines == 1:
            axes = axes.reshape(1, -1)
        if n_sensors == 1:
            axes = axes.reshape(-1, 1)
        
        for i, engine_id in enumerate(sample_engines):
            engine_data = self.train_data[self.train_data['unit_id'] == engine_id]
            for j, sensor in enumerate(selected_sensors):
                axes[i, j].plot(engine_data['time_cycles'], engine_data[sensor])
                axes[i, j].set_title(f'Engine {engine_id} - {sensor}')
                axes[i, j].set_xlabel('Time Cycles')
                axes[i, j].set_ylabel('Sensor Value')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensor trends visualization saved to {save_path}")
        plt.close()

