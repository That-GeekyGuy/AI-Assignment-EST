# Turbofan Engine RUL Prediction - Rolls-Royce Aerospace

## Project Overview

This project implements a deep learning solution for predicting the **Remaining Useful Life (RUL)** of turbofan aircraft engines using multi-sensor time-series data. The system enables proactive maintenance scheduling for Rolls-Royce Aerospace, reducing downtime and maintenance costs.

## Problem Statement

Aircraft engines degrade over time due to operational wear and tear. Predicting the RUL allows maintenance teams to:
- Schedule maintenance proactively before failure
- Optimize maintenance costs
- Reduce unplanned downtime
- Improve safety and reliability

## Dataset

The project uses the **NASA Turbofan Engine Degradation Simulation Dataset** (FD001 subset), which can be obtained from:
- **Kaggle**: [NASA Turbofan Engine Degradation Simulation Data](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- **GitHub**: [CMAPSS Dataset](https://github.com/Lucky-Loek/CMAPSS_data)

### Dataset Structure

The project supports 4 datasets (FD001, FD002, FD003, FD004) with different characteristics:

- **FD001**: 100 training engines, 1 condition, 1 fault mode (HPC Degradation)
- **FD002**: 260 training engines, 6 conditions, 1 fault mode (HPC Degradation)
- **FD003**: 100 training engines, 1 condition, 2 fault modes (HPC + Fan Degradation)
- **FD004**: 248 training engines, 6 conditions, 2 fault modes (HPC + Fan Degradation)

Each dataset contains:
- **Training Data** (`train_FD00X.txt`): Contains run-to-failure data
  - Column 1: Unit number (engine ID)
  - Column 2: Time in cycles
  - Columns 3-5: Operational settings (3 features)
  - Columns 6-26: Sensor measurements (21 sensors)

- **Test Data** (`test_FD00X.txt`): Contains time-series data until a certain point
- **RUL Data** (`RUL_FD00X.txt`): Contains the actual RUL for each engine in the test set

## Project Structure

```
.
├── CMAPSSData/                    # Dataset directory (contains all 4 datasets)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD002.txt
│   ├── test_FD002.txt
│   ├── RUL_FD002.txt
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   ├── RUL_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   ├── RUL_FD004.txt
│   └── readme.txt
├── models/                        # Saved models
├── visualizations/                # Generated plots and diagrams
├── data_preprocessing.py          # Data loading and preprocessing
├── model.py                       # LSTM/RNN model implementation
├── evaluation.py                  # Evaluation metrics and visualization
├── main.py                        # Main training script
├── streamlit_app.py               # Streamlit GUI application
├── run_streamlit.py               # Streamlit launcher script
├── example_usage.py               # Example usage script
├── download_data.py               # Data verification utility
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── STREAMLIT_README.md            # Streamlit GUI documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the dataset**:
   - The dataset should be in the `CMAPSSData/` directory
   - Verify all files are present: `python download_data.py verify`
   - If missing, download from Kaggle or GitHub and extract to `CMAPSSData/` folder

## Usage

### 🎥 Streamlit GUI Demo

Watch the interactive Streamlit interface in action:

Arc 2025-11-08 23-31-52.mp4


## Pipeline Overview

### Flow Diagram

The project includes a comprehensive flow diagram (`visualizations/flow_diagram.png`) showing:

1. **Data Loading**: Load training and test datasets
2. **Data Preprocessing**: 
   - Feature selection
   - Missing value handling
   - RUL calculation
   - Feature scaling (MinMaxScaler)
   - Sequence creation (sliding window)
3. **Data Visualization**: 
   - RUL distribution
   - Sensor trends
   - Correlation analysis
4. **Model Training**: 
   - LSTM/Bidirectional LSTM/Stacked LSTM
   - Early stopping
   - Learning rate reduction
5. **Model Evaluation**: 
   - Metrics calculation
   - Prediction visualization
   - Error analysis
6. **Results**: 
   - RUL predictions
   - Maintenance scheduling recommendations

## Data Preprocessing

### Steps

1. **Data Loading**: Load training and test data from text files
2. **RUL Calculation**: Calculate RUL for training data as `RUL = Max Cycles - Current Cycles`
3. **Feature Selection**: Select sensor measurements (21 sensors) or include operational settings
4. **Feature Scaling**: Normalize features using MinMaxScaler (range 0-1)
5. **Sequence Creation**: Create sequences using sliding window approach
   - Default sequence length: 50 cycles
   - Each sequence predicts the RUL at the end of the sequence

### Visualizations

The preprocessing module generates:
- **Data Distribution**: RUL histogram, engine lifespan distribution
- **Sensor Trends**: Time-series plots of sensor measurements
- **Correlation Heatmap**: Correlation between sensors

## Model Architecture

### LSTM Variants

1. **Simple LSTM**: Single LSTM layer with dropout
2. **Bidirectional LSTM**: Bidirectional LSTM layer (captures past and future context)
3. **Stacked LSTM**: Multiple LSTM layers for deeper learning

### Architecture Details

- **Input Layer**: Sequences of sensor data (sequence_length × n_features)
- **LSTM Layer(s)**: 64 units (configurable)
- **Dropout**: 0.2 (regularization)
- **Dense Layers**: 
  - Hidden layer 1: 50 units (ReLU)
  - Hidden layer 2: 25 units (ReLU)
  - Output layer: 1 unit (Linear) - RUL prediction

### Training

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MAE, MSE
- **Callbacks**:
  - Early Stopping: Stop if validation loss doesn't improve for 15 epochs
  - Learning Rate Reduction: Reduce LR by 50% if validation loss plateaus
  - Model Checkpoint: Save best model based on validation loss

## Evaluation Metrics

### Metrics Used

1. **RMSE (Root Mean Squared Error)**: Measures average prediction error
2. **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals
3. **R² Score**: Coefficient of determination (measures goodness of fit)
4. **PHM08 Score**: Challenge-specific scoring function
   - Penalizes underestimation more than overestimation
   - Formula: `exp(-a) - 1` if under-estimated, `exp(a) - 1` if over-estimated
   - where `a = (actual - predicted) / 13`

### Visualizations

- **Predictions vs Actual**: Scatter plot and residual plot
- **Error Distribution**: Histogram of prediction errors
- **Training History**: Loss and MAE curves over epochs
- **Engine Predictions**: Bar charts for specific engines

## Results

After training, the following outputs are generated:

1. **Saved Model**: `models/best_rul_model.h5` (best model based on validation loss)
2. **Visualizations**:
   - `data_distribution.png`: Data characteristics
   - `sensor_trends.png`: Sensor measurements over time
   - `predictions_vs_actual.png`: Prediction accuracy
   - `error_distribution.png`: Error analysis
   - `training_history.png`: Training curves
   - `flow_diagram.png`: Complete pipeline diagram

## Performance Considerations

### Expected Performance

- **RMSE**: Typically 15-25 cycles
- **MAE**: Typically 10-20 cycles
- **R² Score**: Typically 0.85-0.95

### Improvements

To improve model performance:
1. **Feature Engineering**: Create derived features (moving averages, trends)
2. **Hyperparameter Tuning**: Optimize sequence length, LSTM units, learning rate
3. **Ensemble Methods**: Combine multiple models
4. **Advanced Architectures**: Attention mechanisms, Transformer models
5. **Data Augmentation**: Generate synthetic sequences

## Maintenance Scheduling

The predicted RUL values can be used to:
1. **Threshold-Based Scheduling**: Schedule maintenance when RUL < threshold
2. **Risk-Based Scheduling**: Consider operational criticality
3. **Cost Optimization**: Balance maintenance costs vs. failure risks

## Future Work

1. **Multi-Engine Models**: Predict RUL for multiple engine types
2. **Real-Time Prediction**: Deploy model for real-time monitoring
3. **Uncertainty Quantification**: Provide confidence intervals for predictions
4. **Explainability**: Understand which sensors contribute most to predictions
5. **Transfer Learning**: Adapt model to new engine types

## References

1. Saxena, A., et al. "Damage propagation modeling for aircraft engine run-to-failure simulation." International Conference on Prognostics and Health Management (2008).
2. NASA Prognostics Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
3. Kaggle Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## License

This project is for educational and research purposes.

## Author

Rolls-Royce Aerospace - Predictive Maintenance Project

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Note**: This project is part of an AI assignment and demonstrates best practices in time-series forecasting, deep learning, and predictive maintenance.


