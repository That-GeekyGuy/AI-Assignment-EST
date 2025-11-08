"""
Example usage script for RUL Prediction
This script demonstrates how to use the RUL prediction pipeline
"""

from data_preprocessing import TurbofanDataPreprocessor
from model import RULPredictor
from evaluation import RULEvaluator
import numpy as np

def example_basic_usage():
    """Basic example of using the RUL prediction pipeline"""
    
    print("="*70)
    print("EXAMPLE: Basic RUL Prediction Usage")
    print("="*70)
    print()
    
    # Step 1: Initialize preprocessor
    print("Step 1: Initializing data preprocessor...")
    preprocessor = TurbofanDataPreprocessor(data_path='CMAPSSData')
    
    # Step 2: Load data
    print("Step 2: Loading data...")
    try:
        train_data, test_data, test_rul = preprocessor.load_data(
            train_file='train_FD001.txt',
            test_file='test_FD001.txt',
            rul_file='RUL_FD001.txt'
        )
        print("✓ Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure the data files are in the 'CMAPSSData/' directory")
        print("Run: python download_data.py verify")
        return
    
    # Step 3: Prepare training data
    print("Step 3: Preparing training data...")
    X_train, y_train, feature_cols = preprocessor.prepare_train_data(use_sensors_only=True)
    
    # Step 4: Scale features
    print("Step 4: Scaling features...")
    X_train_scaled, _ = preprocessor.scale_features(X_train, X_train[:10])  # Dummy test data
    
    # Step 5: Create sequences
    print("Step 5: Creating sequences...")
    sequence_length = 50
    X_train_seq, y_train_seq = preprocessor.create_sequences(
        X_train_scaled, y_train,
        sequence_length=sequence_length,
        step_size=1
    )
    print(f"✓ Created {len(X_train_seq)} sequences")
    print()
    
    # Step 6: Build and compile model
    print("Step 6: Building model...")
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = RULPredictor(
        input_shape=input_shape,
        model_type='bidirectional_lstm',
        units=64,
        dropout_rate=0.2
    )
    model.compile_model(learning_rate=0.001)
    print("✓ Model built and compiled!")
    print()
    
    # Step 7: Split data for training
    print("Step 7: Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_seq, y_train_seq, test_size=0.2, random_state=42
    )
    print(f"✓ Training set: {X_train_final.shape}")
    print(f"✓ Validation set: {X_val.shape}")
    print()
    
    # Step 8: Train model (short training for example)
    print("Step 8: Training model (5 epochs for demo)...")
    history = model.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=5,  # Short training for example
        batch_size=32,
        verbose=1
    )
    print("✓ Model training completed!")
    print()
    
    # Step 9: Evaluate on validation set
    print("Step 9: Evaluating model...")
    evaluator = RULEvaluator()
    y_pred = model.predict(X_val)
    metrics = evaluator.calculate_metrics(y_val, y_pred)
    evaluator.print_metrics()
    
    print("="*70)
    print("Example completed successfully!")
    print("="*70)

def example_visualization_only():
    """Example of generating visualizations without training"""
    
    print("="*70)
    print("EXAMPLE: Data Visualization")
    print("="*70)
    print()
    
    preprocessor = TurbofanDataPreprocessor(data_path='CMAPSSData')
    
    try:
        train_data, test_data, test_rul = preprocessor.load_data(
            train_file='train_FD001.txt',
            test_file='test_FD001.txt',
            rul_file='RUL_FD001.txt'
        )
        print("✓ Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print("Generating visualizations...")
    preprocessor.visualize_data_distribution()
    preprocessor.visualize_sensor_trends(n_engines=3, n_sensors=6)
    print("✓ Visualizations saved to 'visualizations/' directory")
    print()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'viz':
        example_visualization_only()
    else:
        example_basic_usage()


