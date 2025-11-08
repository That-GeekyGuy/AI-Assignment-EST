# Project Summary: Turbofan Engine RUL Prediction

## Overview

This project implements a **deep learning-based predictive maintenance system** for Rolls-Royce Aerospace to predict the Remaining Useful Life (RUL) of turbofan aircraft engines using multi-sensor time-series data.

## Key Features

### 1. Data Preprocessing
- **Data Loading**: Handles NASA CMAPSS dataset format
- **RUL Calculation**: Computes RUL from time-series data
- **Feature Engineering**: Selects and scales sensor measurements
- **Sequence Creation**: Creates time-series sequences for LSTM input
- **Visualization**: Generates data distribution and sensor trend plots

### 2. Deep Learning Models
- **LSTM**: Standard Long Short-Term Memory network
- **Bidirectional LSTM**: Captures past and future context
- **Stacked LSTM**: Multiple LSTM layers for deeper learning
- **Regularization**: Dropout layers to prevent overfitting
- **Early Stopping**: Prevents overfitting during training

### 3. Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **PHM08 Score**: Challenge-specific scoring function
- **Visualizations**: Prediction plots, error distributions, training history

### 4. Flow Diagram
- Comprehensive pipeline visualization
- Shows preprocessing steps
- Documents visualization steps
- Illustrates model training and evaluation flow

## Project Structure

```
.
├── CMAPSSData/              # Dataset directory (contains FD001-FD004)
├── models/                  # Saved models
├── visualizations/          # Generated plots
├── data_preprocessing.py    # Data loading and preprocessing
├── model.py                 # LSTM/RNN model implementation
├── evaluation.py            # Evaluation metrics and visualization
├── main.py                  # Main training script
├── example_usage.py         # Example usage script
├── download_data.py         # Data verification utility
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── PROJECT_SUMMARY.md       # This file
```

## Technical Approach

### Data Pipeline
1. **Load Data**: Read training and test data from text files
2. **Calculate RUL**: RUL = Max Cycles - Current Cycles
3. **Feature Selection**: Use 21 sensor measurements
4. **Normalization**: MinMaxScaler (0-1 range)
5. **Sequence Creation**: Sliding window with length 50
6. **Train/Val Split**: 80/20 split

### Model Architecture
- **Input**: Sequences of sensor data (50 × 21)
- **LSTM Layer**: 64 units (bidirectional)
- **Dropout**: 0.2
- **Dense Layers**: 50 → 25 → 1
- **Output**: RUL prediction (single value)

### Training Strategy
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Callbacks**: Early stopping, LR reduction, model checkpoint
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)

## Performance Metrics

### Expected Results
- **RMSE**: 15-25 cycles
- **MAE**: 10-20 cycles
- **R² Score**: 0.85-0.95

### Evaluation Approach
- Test on unseen engine data
- Compare predicted vs. actual RUL
- Analyze error distribution
- Generate maintenance scheduling recommendations

## Usage

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify dataset (should be in CMAPSSData/ directory)
python download_data.py verify

# 3. Run training (default: FD001 dataset)
python main.py --create_diagram

# 4. View results
# - Models: models/
# - Visualizations: visualizations/
```

### Customization
```bash
# Use different dataset (FD001, FD002, FD003, or FD004)
python main.py --dataset FD002 --create_diagram

# Custom parameters
python main.py --dataset FD001 --sequence_length 60 --epochs 150 --model_type stacked_lstm --units 128
```

## Deliverables

### Code Files
- ✅ Data preprocessing module
- ✅ LSTM/RNN model implementation
- ✅ Evaluation module
- ✅ Main training script
- ✅ Example usage script
- ✅ Data download utility

### Documentation
- ✅ README with comprehensive documentation
- ✅ Flow diagram (visualizations/flow_diagram.png)
- ✅ Code comments and docstrings
- ✅ Project summary

### Visualizations
- ✅ Data distribution plots
- ✅ Sensor trend plots
- ✅ Training history plots
- ✅ Prediction vs. actual plots
- ✅ Error distribution plots
- ✅ Flow diagram

## Key Achievements

1. **Complete Pipeline**: End-to-end solution from data loading to prediction
2. **Multiple Model Types**: Support for different LSTM architectures
3. **Comprehensive Evaluation**: Multiple metrics and visualizations
4. **Well Documented**: Clear documentation and code comments
5. **Flow Diagram**: Visual representation of the entire pipeline
6. **Modular Design**: Easy to extend and modify

## Future Enhancements

1. **Feature Engineering**: Create derived features (moving averages, trends)
2. **Hyperparameter Tuning**: Optimize sequence length, units, learning rate
3. **Ensemble Methods**: Combine multiple models
4. **Attention Mechanisms**: Add attention layers for better interpretability
5. **Real-Time Deployment**: Deploy model for real-time monitoring
6. **Uncertainty Quantification**: Provide confidence intervals
7. **Explainability**: Understand sensor contributions to predictions

## References

- NASA Prognostics Data Repository
- CMAPSS Dataset (Kaggle/GitHub)
- PHM08 Challenge
- Deep Learning for Time-Series Forecasting

## Conclusion

This project provides a complete solution for predicting aircraft engine RUL using deep learning. The modular design, comprehensive evaluation, and detailed documentation make it suitable for both research and practical applications in predictive maintenance.

---

**Project Status**: ✅ Complete
**Last Updated**: 2024


