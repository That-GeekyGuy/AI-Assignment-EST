"""
Deep Learning Model for RUL Prediction
Implements LSTM/RNN architectures for time-series prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

class RULPredictor:
    """LSTM/RNN model for predicting Remaining Useful Life"""
    
    def __init__(self, input_shape, model_type='lstm', units=64, dropout_rate=0.2):
        """
        Initialize the RUL predictor model
        
        Args:
            input_shape: Tuple (sequence_length, n_features)
            model_type: 'lstm', 'bidirectional_lstm', or 'stacked_lstm'
            units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build the LSTM/RNN model architecture"""
        model = Sequential()
        
        if self.model_type == 'lstm':
            # Simple LSTM model
            model.add(LSTM(units=self.units, 
                          input_shape=self.input_shape,
                          return_sequences=False))
            model.add(Dropout(self.dropout_rate))
            
        elif self.model_type == 'bidirectional_lstm':
            # Bidirectional LSTM
            model.add(Bidirectional(LSTM(units=self.units,
                                        input_shape=self.input_shape,
                                        return_sequences=False)))
            model.add(Dropout(self.dropout_rate))
            
        elif self.model_type == 'stacked_lstm':
            # Stacked LSTM layers
            model.add(LSTM(units=self.units,
                          input_shape=self.input_shape,
                          return_sequences=True))
            model.add(Dropout(self.dropout_rate))
            model.add(LSTM(units=self.units // 2,
                          return_sequences=False))
            model.add(Dropout(self.dropout_rate))
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Dense layers for regression
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1, activation='linear'))  # Linear activation for regression
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001, loss='mse', metrics=None):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            self.build_model()
        
        if metrics is None:
            metrics = ['mae', 'mse']
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Build the model explicitly to ensure it's ready for summary
        # This is needed for some Keras versions
        if not self.model.built:
            self.model.build(input_shape=(None,) + self.input_shape)
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training RUL values
            X_val: Validation sequences
            y_val: Validation RUL values
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('models/best_rul_model.h5', monitor='val_loss', 
                          save_best_only=True, verbose=0)  # Set to 0 to reduce output
        ]
        
        os.makedirs('models', exist_ok=True)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load_model() first.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath='models/rul_model.h5'):
        """Save the model to file"""
        if self.model is None:
            raise ValueError("Model not built. Nothing to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/rul_model.h5'):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or compile_model() first.")
        
        # Ensure model is built before getting summary
        # Check if model is built, if not, build it
        try:
            # Check if model has 'built' attribute
            if hasattr(self.model, 'built'):
                if not self.model.built:
                    self.model.build(input_shape=(None,) + self.input_shape)
            else:
                # For older Keras versions or if built attribute doesn't exist
                # Try to build anyway - it's safe to call multiple times
                try:
                    self.model.build(input_shape=(None,) + self.input_shape)
                except:
                    pass  # Model might already be built or will be built during fit
        
        except Exception as e:
            # If building fails, try to get summary anyway
            # It might work if model was built during compilation
            pass
        
        # Get and print summary
        try:
            return self.model.summary()
        except ValueError as e:
            # If summary still fails, print a message
            print(f"Note: Could not display model summary at this stage: {e}")
            print("Model will be built automatically during training.")
            print(f"Model architecture: {self.model_type} LSTM with {self.units} units")
            print(f"Input shape: {self.input_shape}")
            return None

