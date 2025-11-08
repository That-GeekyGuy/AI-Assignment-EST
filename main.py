"""
Main Script for Turbofan Engine RUL Prediction
Rolls-Royce Aerospace - Predictive Maintenance Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_preprocessing import TurbofanDataPreprocessor
from model import RULPredictor
from evaluation import RULEvaluator
import os
import argparse

def create_flow_diagram(save_path='visualizations/flow_diagram.png'):
    """
    Create a flow diagram showing the preprocessing and visualization steps
    """
    os.makedirs('visualizations', exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    data_color = '#E3F2FD'  # Light blue
    process_color = '#FFF9C4'  # Light yellow
    model_color = '#C8E6C9'  # Light green
    eval_color = '#FFCCBC'  # Light orange
    viz_color = '#F3E5F5'  # Light purple
    
    # Title
    ax.text(5, 11.5, 'RUL Prediction Pipeline - Flow Diagram', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Step 1: Data Loading
    rect1 = mpatches.FancyBboxPatch((0.5, 9.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.5, 10.2, 'Data Loading', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 9.8, 'train_FD001.txt\ntest_FD001.txt\nRUL_FD001.txt', 
            ha='center', fontsize=9)
    
    # Arrow 1
    ax.annotate('', xy=(2.5, 10), xytext=(3, 10),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 2: Data Preprocessing
    rect2 = mpatches.FancyBboxPatch((3, 9.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(4, 10.2, 'Data Preprocessing', ha='center', fontsize=12, fontweight='bold')
    ax.text(4, 9.8, 'Feature Selection\nMissing Value Handling', 
            ha='center', fontsize=9)
    
    # Arrow 2
    ax.annotate('', xy=(5, 10), xytext=(5.5, 10),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 3: RUL Calculation
    rect3 = mpatches.FancyBboxPatch((5.5, 9.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(6.5, 10.2, 'RUL Calculation', ha='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 9.8, 'Max Cycles - Current Cycles', 
            ha='center', fontsize=9)
    
    # Arrow 3 (down)
    ax.annotate('', xy=(7.5, 9.5), xytext=(7.5, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 4: Feature Scaling
    rect4 = mpatches.FancyBboxPatch((6.5, 7.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(7.5, 8.2, 'Feature Scaling', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 7.8, 'MinMaxScaler\nNormalization', 
            ha='center', fontsize=9)
    
    # Arrow 4 (down)
    ax.annotate('', xy=(7.5, 7.5), xytext=(7.5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 5: Sequence Creation
    rect5 = mpatches.FancyBboxPatch((6.5, 5.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(7.5, 6.2, 'Sequence Creation', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 5.8, 'Sliding Window\nSequence Length: 50', 
            ha='center', fontsize=9)
    
    # Arrow 5 (down)
    ax.annotate('', xy=(7.5, 5.5), xytext=(7.5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 6: Train/Val Split
    rect6 = mpatches.FancyBboxPatch((6.5, 3.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect6)
    ax.text(7.5, 4.2, 'Train/Val Split', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 3.8, '80% Train\n20% Validation', 
            ha='center', fontsize=9)
    
    # Arrow 6 (down)
    ax.annotate('', xy=(7.5, 3.5), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 7: Model Training
    rect7 = mpatches.FancyBboxPatch((6.5, 1.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect7)
    ax.text(7.5, 2.2, 'LSTM Model Training', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 1.8, 'Bidirectional LSTM\nEarly Stopping', 
            ha='center', fontsize=9)
    
    # Arrow 7 (left)
    ax.annotate('', xy=(6.5, 2), xytext=(5.5, 2),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 8: Model Evaluation
    rect8 = mpatches.FancyBboxPatch((3.5, 1.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=eval_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect8)
    ax.text(4.5, 2.2, 'Model Evaluation', ha='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 1.8, 'RMSE, MAE, R²\nPHM08 Score', 
            ha='center', fontsize=9)
    
    # Arrow 8 (left)
    ax.annotate('', xy=(3.5, 2), xytext=(2.5, 2),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Step 9: Predictions
    rect9 = mpatches.FancyBboxPatch((0.5, 1.5), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=eval_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect9)
    ax.text(1.5, 2.2, 'RUL Predictions', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 1.8, 'Test Set Predictions\nMaintenance Scheduling', 
            ha='center', fontsize=9)
    
    # Visualization branches (from preprocessing)
    # Branch 1: Data Distribution
    rect_v1 = mpatches.FancyBboxPatch((0.5, 7.5), 2, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=viz_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_v1)
    ax.text(1.5, 8.2, 'Data Distribution', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 7.8, 'RUL Histogram\nEngine Lifespan', ha='center', fontsize=8)
    
    # Arrow to visualization 1
    ax.annotate('', xy=(2.5, 8), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--', color='gray'))
    
    # Branch 2: Sensor Trends
    rect_v2 = mpatches.FancyBboxPatch((0.5, 5.5), 2, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=viz_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_v2)
    ax.text(1.5, 6.2, 'Sensor Trends', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 5.8, 'Time Series Plots\nCorrelation Heatmap', ha='center', fontsize=8)
    
    # Arrow to visualization 2
    ax.annotate('', xy=(2.5, 6), xytext=(3, 6.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--', color='gray'))
    
    # Branch 3: Training History
    rect_v3 = mpatches.FancyBboxPatch((0.5, 3.5), 2, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=viz_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_v3)
    ax.text(1.5, 4.2, 'Training History', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 3.8, 'Loss Curves\nMAE Trends', ha='center', fontsize=8)
    
    # Arrow to visualization 3
    ax.annotate('', xy=(2.5, 4), xytext=(5.5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--', color='gray'))
    
    # Branch 4: Prediction Results
    rect_v4 = mpatches.FancyBboxPatch((0.5, 0), 2, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=viz_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_v4)
    ax.text(1.5, 0.7, 'Prediction Results', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 0.3, 'Pred vs Actual\nError Distribution', ha='center', fontsize=8)
    
    # Arrow to visualization 4
    ax.annotate('', xy=(1.5, 1.5), xytext=(1.5, 1),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--', color='gray'))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=data_color, edgecolor='black', label='Data'),
        mpatches.Patch(facecolor=process_color, edgecolor='black', label='Preprocessing'),
        mpatches.Patch(facecolor=model_color, edgecolor='black', label='Model'),
        mpatches.Patch(facecolor=eval_color, edgecolor='black', label='Evaluation'),
        mpatches.Patch(facecolor=viz_color, edgecolor='black', label='Visualization')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Flow diagram saved to {save_path}")
    plt.close()

def main():
    """Main function to run the RUL prediction pipeline"""
    parser = argparse.ArgumentParser(description='Turbofan Engine RUL Prediction')
    parser.add_argument('--data_path', type=str, default='CMAPSSData', 
                       help='Path to data directory (default: CMAPSSData)')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use (FD001, FD002, FD003, or FD004)')
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='Sequence length for LSTM input')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='bidirectional_lstm',
                       choices=['lstm', 'bidirectional_lstm', 'stacked_lstm'],
                       help='Type of LSTM model to use')
    parser.add_argument('--units', type=int, default=64,
                       help='Number of LSTM units')
    parser.add_argument('--create_diagram', action='store_true',
                       help='Create flow diagram')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROLLS-ROYCE AEROSPACE - TURBOFAN ENGINE RUL PREDICTION")
    print("="*70)
    print()
    
    # Create flow diagram
    if args.create_diagram:
        print("Creating flow diagram...")
        create_flow_diagram()
        print()
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    print("-"*70)
    print(f"Using dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print()
    
    preprocessor = TurbofanDataPreprocessor(data_path=args.data_path)
    
    # Load data for specified dataset
    train_file = f'train_{args.dataset}.txt'
    test_file = f'test_{args.dataset}.txt'
    rul_file = f'RUL_{args.dataset}.txt'
    
    train_data, test_data, test_rul = preprocessor.load_data(
        train_file=train_file,
        test_file=test_file,
        rul_file=rul_file
    )
    print()
    
    # Prepare training data
    X_train, y_train, feature_cols = preprocessor.prepare_train_data(use_sensors_only=True)
    
    # Prepare test data
    X_test, y_test = preprocessor.prepare_test_data(feature_cols)
    print()
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    print()
    
    # Create sequences
    print(f"Creating sequences with length {args.sequence_length}...")
    X_train_seq, y_train_seq = preprocessor.create_sequences(
        X_train_scaled, y_train, 
        sequence_length=args.sequence_length, 
        step_size=1
    )
    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Training RUL shape: {y_train_seq.shape}")
    print()
    
    # Train/validation split
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_seq, y_train_seq, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train_final.shape}")
    print(f"Validation set: {X_val.shape}")
    print()
    
    # Step 2: Data Visualization
    print("Step 2: Data Visualization")
    print("-"*70)
    preprocessor.visualize_data_distribution()
    preprocessor.visualize_sensor_trends(n_engines=3, n_sensors=6)
    print()
    
    # Step 3: Model Training
    print("Step 3: Model Training")
    print("-"*70)
    input_shape = (X_train_final.shape[1], X_train_final.shape[2])
    model = RULPredictor(
        input_shape=input_shape,
        model_type=args.model_type,
        units=args.units,
        dropout_rate=0.2
    )
    
    model.compile_model(learning_rate=0.001)
    model.get_model_summary()
    print()
    
    # Train model
    print("Training model...")
    history = model.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    print()
    
    # Step 4: Model Evaluation
    print("Step 4: Model Evaluation")
    print("-"*70)
    evaluator = RULEvaluator()
    
    # For test data, we need to create sequences from the last N cycles of each engine
    # Use the last sequence_length cycles of each engine to predict final RUL
    X_test_sequences = []
    y_test_sequences = []
    
    # Get unique engine IDs and sort them
    unique_engines = sorted(test_data['unit_id'].unique())
    
    # Get last sequence_length cycles for each test engine
    for idx, unit_id in enumerate(unique_engines):
        unit_data = test_data[test_data['unit_id'] == unit_id]
        unit_features = unit_data[feature_cols].values
        unit_features_scaled = preprocessor.scaler.transform(unit_features)
        
        if len(unit_features_scaled) >= args.sequence_length:
            # Use the last sequence_length cycles
            X_test_sequences.append(unit_features_scaled[-args.sequence_length:])
        else:
            # Pad with zeros if sequence is shorter than required
            padded = np.zeros((args.sequence_length, unit_features_scaled.shape[1]))
            padded[-len(unit_features_scaled):] = unit_features_scaled
            X_test_sequences.append(padded)
        
        # Get the RUL for this engine (RUL file is ordered by engine ID)
        unit_rul = test_rul.iloc[idx]['RUL']
        y_test_sequences.append(unit_rul)
    
    X_test_seq = np.array(X_test_sequences)
    y_test_seq = np.array(y_test_sequences)
    
    print(f"Test sequences shape: {X_test_seq.shape}")
    print(f"Test RUL shape: {y_test_seq.shape}")
    print()
    
    # Make predictions
    y_pred = model.predict(X_test_seq)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test_seq, y_pred)
    evaluator.print_metrics()
    
    # Visualizations
    evaluator.plot_predictions_vs_actual(y_test_seq, y_pred)
    evaluator.plot_error_distribution(y_test_seq, y_pred)
    evaluator.plot_training_history(history)
    print()
    
    # Step 5: Save model
    print("Step 5: Saving Model")
    print("-"*70)
    model.save_model('models/rul_model_final.h5')
    print()
    
    print("="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved in:")
    print("  - Models: models/")
    print("  - Visualizations: visualizations/")
    print()

if __name__ == '__main__':
    main()

