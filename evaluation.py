"""
Evaluation Module for RUL Prediction
Contains metrics and visualization functions for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class RULEvaluator:
    """Class for evaluating RUL prediction models"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays and flatten if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure predictions are non-negative (RUL cannot be negative)
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Custom metric: Score function (used in PHM08 challenge)
        # Penalty: exp(-a) if under-estimated, exp(a) - 1 if over-estimated
        # where a = (actual - predicted) / 13
        score = 0
        for i in range(len(y_true)):
            a = (y_true[i] - y_pred[i]) / 13.0
            if y_pred[i] < y_true[i]:  # Under-estimation
                score += np.exp(-a) - 1
            else:  # Over-estimation
                score += np.exp(a) - 1
        
        # Ensure all metrics are scalars (not arrays)
        self.metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2_Score': float(r2),
            'PHM08_Score': float(score)
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        if not self.metrics:
            print("No metrics calculated. Call calculate_metrics() first.")
            return
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        for metric, value in self.metrics.items():
            # Handle NaN and Inf values
            if np.isnan(value) or np.isinf(value):
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
        print("="*50 + "\n")
    
    def plot_predictions_vs_actual(self, y_true, y_pred, save_path='visualizations/predictions_vs_actual.png'):
        """
        Plot predicted vs actual RUL values
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            save_path: Path to save the plot
        """
        os.makedirs('visualizations', exist_ok=True)
        
        # Convert to numpy arrays and flatten if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual RUL (Cycles)')
        axes[0].set_ylabel('Predicted RUL (Cycles)')
        axes[0].set_title('Predicted vs Actual RUL')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred.flatten()
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted RUL (Cycles)')
        axes[1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions vs actual plot saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, save_path='visualizations/error_distribution.png'):
        """
        Plot error distribution
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            save_path: Path to save the plot
        """
        os.makedirs('visualizations', exist_ok=True)
        
        # Convert to numpy arrays and flatten if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0)
        
        errors = y_true - y_pred
        absolute_errors = np.abs(errors)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error distribution
        axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Error (Actual - Predicted)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Absolute error distribution
        axes[1].hist(absolute_errors, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Absolute Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
        plt.close()
    
    def plot_training_history(self, history, save_path='visualizations/training_history.png'):
        """
        Plot training history (loss and metrics over epochs)
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save the plot
        """
        os.makedirs('visualizations', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss', marker='o')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE', marker='o')
            axes[1].plot(history.history['val_mae'], label='Validation MAE', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error Over Epochs')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def plot_engine_predictions(self, engine_ids, y_true_dict, y_pred_dict, 
                               save_path='visualizations/engine_predictions.png'):
        """
        Plot RUL predictions for specific engines
        
        Args:
            engine_ids: List of engine IDs to plot
            y_true_dict: Dictionary mapping engine_id to true RUL
            y_pred_dict: Dictionary mapping engine_id to predicted RUL
            save_path: Path to save the plot
        """
        os.makedirs('visualizations', exist_ok=True)
        
        n_engines = len(engine_ids)
        fig, axes = plt.subplots(n_engines, 1, figsize=(12, 4*n_engines))
        
        if n_engines == 1:
            axes = [axes]
        
        for i, engine_id in enumerate(engine_ids):
            true_rul = y_true_dict[engine_id]
            pred_rul = max(y_pred_dict[engine_id], 0)  # Ensure non-negative
            
            axes[i].barh(['Actual', 'Predicted'], [true_rul, pred_rul], 
                        color=['blue', 'orange'], alpha=0.7)
            axes[i].set_xlabel('RUL (Cycles)')
            axes[i].set_title(f'Engine {engine_id} - RUL Prediction')
            axes[i].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            axes[i].text(true_rul, 0, f' {true_rul:.1f}', 
                        va='center', fontweight='bold')
            axes[i].text(pred_rul, 1, f' {pred_rul:.1f}', 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Engine predictions plot saved to {save_path}")
        plt.close()


