"""
Streamlit GUI for Turbofan Engine RUL Prediction
Rolls-Royce Aerospace - Predictive Maintenance Project
Modern UI with gradients, squircles, and tabs
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
from pathlib import Path
import warnings
from datetime import datetime
import shutil
warnings.filterwarnings('ignore')

# Import project modules
from data_preprocessing import TurbofanDataPreprocessor
from model import RULPredictor
from evaluation import RULEvaluator

# Page configuration
st.set_page_config(
    page_title="Turbofan Engine RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Rolls-Royce Aerospace - Predictive Maintenance System"
    }
)

# Custom CSS for black and white theme with professional styling
st.markdown("""
    <style>
    /* Import minimalist font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');
    
    /* Global font styling - thin and minimalist */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 300;
    }
    
    /* Main background - pure black */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        background-attachment: fixed;
    }
    
    /* Main container with dark background */
    .main .block-container {
        background: #0a0a0a;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
        border: 1px solid #2a2a2a;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Header with light gradient text */
    .main-header {
        font-size: 3rem;
        font-weight: 200;
        color: #ffffff;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(255, 255, 255, 0.1);
    }
    
    /* Professional buttons - black and white theme with better contrast */
    .stButton>button {
        background: #ffffff;
        color: #000000 !important;
        border: 2px solid #ffffff;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 400;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: #000000;
        color: #ffffff !important;
        border: 2px solid #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 255, 255, 0.4);
    }
    
    .stButton>button:active {
        color: #000000 !important;
    }
    
    /* Sidebar styling - dark */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 100%);
        border-right: 1px solid #2a2a2a;
    }
    
    /* Sidebar text - light */
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    
    /* Tabs styling - black and white theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 10px 20px;
        border: none;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
        color: #606060;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #ffffff;
        border-radius: 0;
        border: none;
        border-bottom: 2px solid #ffffff;
        box-shadow: none;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="false"] {
        color: #909090;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background: transparent;
        color: #c0c0c0;
        border-bottom: 2px solid #606060;
    }
    
    /* Metric containers - light text with thin font */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c0c0c0;
        font-weight: 300;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    [data-testid="stMetricDelta"] {
        color: #d0d0d0;
        font-weight: 300;
    }
    
    /* Info boxes - dark background */
    .info-box {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        color: #e0e0e0;
    }
    
    /* Success/Error messages - black and white theme */
    .stSuccess {
        background: #1a1a1a;
        border-radius: 12px;
        border-left: 4px solid #d0d0d0;
        color: #e0e0e0;
    }
    
    .stError {
        background: #1a1a1a;
        border-radius: 12px;
        border-left: 4px solid #808080;
        color: #c0c0c0;
    }
    
    /* Warning messages */
    .stWarning {
        background: #1a1a1a;
        border-radius: 12px;
        border-left: 4px solid #a0a0a0;
        color: #d0d0d0;
    }
    
    /* Info messages */
    .stInfo {
        background: #1a1a1a;
        border-radius: 12px;
        border-left: 4px solid #ffffff;
        color: #e0e0e0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: #2a2a2a;
    }
    
    .streamlit-expanderContent {
        background: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Selectbox and inputs - black and white theme */
    .stSelectbox>div>div {
        background: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    .stSelectbox>div>div>div {
        color: #e0e0e0;
        font-weight: 300;
    }
    
    .stTextInput>div>div>input {
        background: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    .stTextArea>div>div>textarea {
        background: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    .stNumberInput>div>div>input {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        font-weight: 300;
    }
    
    /* Slider styling - matches theme perfectly */
    .stSlider {
        padding: 2rem 0 1rem 0 !important;
    }
    
    /* Slider label - thin and clean */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 300 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Slider wrapper for spacing */
    .stSlider > div {
        padding: 0.5rem 0 !important;
    }
    
    /* Track background (unfilled) - dark like inputs */
    .stSlider div[data-baseweb="slider"] > div:first-child {
        background: #1a1a1a !important;
        height: 3px !important;
        border-radius: 0 !important;
    }
    
    /* Track fill (filled part) - white accent */
    .stSlider div[data-baseweb="slider"] > div:first-child > div {
        background: #ffffff !important;
        height: 3px !important;
        border-radius: 0 !important;
    }
    
    /* Slider thumb - white square with border */
    .stSlider div[role="slider"] {
        background: #000000 !important;
        border: 2px solid #ffffff !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 0 !important;
        box-shadow: 0 2px 8px rgba(255, 255, 255, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    /* Hover - grows slightly */
    .stSlider div[role="slider"]:hover {
        width: 18px !important;
        height: 18px !important;
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Active/Focus - inverted */
    .stSlider div[role="slider"]:active,
    .stSlider div[role="slider"]:focus {
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.1) !important;
        outline: none !important;
    }
    
    /* Slider value display */
    .stSlider [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
        font-weight: 300 !important;
    }
    
    /* All slider text */
    .stSlider div, .stSlider span, .stSlider p {
        color: #ffffff !important;
        font-weight: 300 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #3a3a3a;
        background: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Progress bar */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #ffffff 0%, #d0d0d0 100%);
    }
    
    /* Headers - light and thin for readability */
    h1 {
        color: #e0e0e0;
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #e0e0e0;
        font-weight: 300;
        letter-spacing: -0.3px;
    }
    
    h3, h4, h5, h6 {
        color: #e0e0e0;
        font-weight: 300;
    }
    
    /* Paragraph text */
    p {
        color: #d0d0d0;
        font-weight: 300;
        line-height: 1.6;
    }
    
    /* Links */
    a {
        color: #ffffff;
        font-weight: 400;
        text-decoration: none;
    }
    
    a:hover {
        color: #ffffff;
        text-decoration: underline;
    }
    
    /* Code blocks */
    code {
        background: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        padding: 2px 6px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure text is readable in all contexts */
    .element-container {
        color: #e0e0e0;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #d0d0d0;
    }
    
    .stMarkdown p {
        color: #d0d0d0;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff;
    }
    
    /* Radio buttons */
    .stRadio>div>label {
        color: #e0e0e0;
        font-weight: 300;
    }
    
    /* Checkbox */
    .stCheckbox>label {
        color: #e0e0e0;
        font-weight: 300;
    }
    
    /* Input labels */
    label {
        font-weight: 300 !important;
        letter-spacing: 0.3px;
    }
    
    /* Number input */
    .stNumberInput>div>div>input {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
    }
    
    /* File uploader */
    .stFileUploader>div {
        background: #1a1a1a;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: #ffffff;
        color: #000000 !important;
        border: 2px solid #ffffff;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .stDownloadButton>button:hover {
        background: #000000;
        color: #ffffff !important;
        border: 2px solid #ffffff;
    }
    
    /* Ensure all text elements are visible */
    .stTextInput label, .stTextArea label, .stSelectbox label,
    .stMultiSelect label, .stNumberInput label, .stDateInput label,
    .stTimeInput label, .stFileUploader label, .stColorPicker label {
        color: #d0d0d0 !important;
    }
    
    /* Ensure divs and spans have visible text */
    div, span, label {
        color: inherit;
    }
    
    /* Make sure captions are visible */
    .caption, small, .stCaption {
        color: #b0b0b0 !important;
    }
    
    /* Ensure text in all components is visible */
    [class*="st"] {
        color: #d0d0d0;
    }
    
    /* Specific override for very dark text */
    * {
        color: inherit;
    }
    
    /* Ensure body text is light */
    body {
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'test_rul' not in st.session_state:
    st.session_state.test_rul = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_artifacts' not in st.session_state:
    st.session_state.model_artifacts = {}
if 'loaded_model_path' not in st.session_state:
    st.session_state.loaded_model_path = None

# Model artifacts management
def save_model_artifact(model, preprocessor, config, metrics=None, history=None):
    """Save model artifact with metadata"""
    os.makedirs('model_artifacts', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = f"model_artifacts/model_{timestamp}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(artifact_dir, "model.h5")
    model.save_model(model_path)
    
    # Save preprocessor scaler
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    
    # Save configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics if available
    if metrics:
        metrics_path = os.path.join(artifact_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save training history if available
    if history:
        history_path = os.path.join(artifact_dir, "history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
    
    return artifact_dir

def load_model_artifact(artifact_dir):
    """Load model artifact"""
    model_path = os.path.join(artifact_dir, "model.h5")
    config_path = os.path.join(artifact_dir, "config.json")
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, config_path, scaler_path]):
        raise ValueError("Invalid artifact directory")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = RULPredictor(
        input_shape=tuple(config['input_shape']),
        model_type=config['model_type'],
        units=config['units'],
        dropout_rate=config['dropout_rate']
    )
    model.load_model(model_path)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, config, scaler

def list_model_artifacts():
    """List all saved model artifacts"""
    artifacts_dir = Path('model_artifacts')
    if not artifacts_dir.exists():
        return []
    
    artifacts = []
    for artifact_dir in artifacts_dir.iterdir():
        if artifact_dir.is_dir():
            config_path = artifact_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                artifacts.append({
                    'path': str(artifact_dir),
                    'name': artifact_dir.name,
                    'config': config,
                    'created': artifact_dir.stat().st_mtime
                })
    
    return sorted(artifacts, key=lambda x: x['created'], reverse=True)

def main():
    # Header with gradient
    st.markdown('<h1 class="main-header">✈️ Turbofan Engine RUL Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #b0b0b0; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 500;">Rolls-Royce Aerospace - Predictive Maintenance System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # Dataset selection
        st.markdown("#### 📊 Dataset")
        dataset = st.selectbox(
            "Select Dataset",
            ["FD001", "FD002", "FD003", "FD004"],
            help="FD001: 100 engines, 1 condition, 1 fault mode\n"
                 "FD002: 260 engines, 6 conditions, 1 fault mode\n"
                 "FD003: 100 engines, 1 condition, 2 fault modes\n"
                 "FD004: 248 engines, 6 conditions, 2 fault modes"
        )
        
        data_path = st.text_input("Data Path", value="CMAPSSData", help="Path to CMAPSSData folder")
        
        st.markdown("---")
        
        # Model configuration
        st.markdown("#### 🤖 Model")
        model_type = st.selectbox(
            "Model Type",
            ["bidirectional_lstm", "lstm", "stacked_lstm"],
            help="Bidirectional LSTM captures past and future context"
        )
        
        sequence_length = st.slider("Sequence Length", 30, 100, 50, help="Length of input sequences")
        lstm_units = st.slider("LSTM Units", 32, 256, 64, step=32)
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.01, 0.0005], index=0)
        
        st.markdown("---")
        
        # Training configuration
        st.markdown("#### 🏋️ Training")
        epochs = st.slider("Epochs", 10, 200, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        
        st.markdown("---")
        
        # Model artifacts
        st.markdown("#### 📦 Model Artifacts")
        artifacts = list_model_artifacts()
        if artifacts:
            artifact_names = [f"{a['name']} ({a['config']['dataset']})" for a in artifacts]
            selected_artifact = st.selectbox("Load Saved Model", ["None"] + artifact_names)
            if selected_artifact != "None":
                artifact_idx = artifact_names.index(selected_artifact)
                if st.button("Load Model Artifact"):
                    try:
                        artifact = artifacts[artifact_idx]
                        model, config, scaler = load_model_artifact(artifact['path'])
                        st.session_state.model = model
                        st.session_state.loaded_model_path = artifact['path']
                        # Update preprocessor scaler
                        if st.session_state.preprocessor:
                            st.session_state.preprocessor.scaler = scaler
                        st.success(f"✅ Model loaded from {artifact['name']}")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
        else:
            st.info("No saved models found")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home", 
        "📊 Data Exploration", 
        "🤖 Model Training", 
        "📦 Model Artifacts",
        "📈 Predictions", 
        "📉 Evaluation"
    ])
    
    with tab1:
        show_home_page(dataset)
    with tab2:
        show_data_exploration(dataset, data_path)
    with tab3:
        show_model_training(dataset, data_path, model_type, sequence_length, 
                          lstm_units, dropout_rate, learning_rate, epochs, batch_size)
    with tab4:
        show_model_artifacts()
    with tab5:
        show_predictions()
    with tab6:
        show_evaluation()

def show_home_page(dataset):
    """Display home page with project overview"""
    st.header("Welcome to Turbofan Engine RUL Prediction System")
    
    # Metrics with gradient cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Dataset", dataset)
        st.metric("Datasets Available", "4")
    
    with col2:
        st.metric("Model Types", "3")
        st.metric("Supported Features", "21 Sensors")
    
    with col3:
        st.metric("Prediction Target", "RUL (Cycles)")
        st.metric("Problem Type", "Time-Series Regression")
    
    st.markdown("---")
    
    # Info boxes with gradients
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📋 Project Overview")
    st.write("""
    This application provides a comprehensive solution for predicting the Remaining Useful Life (RUL) 
    of turbofan aircraft engines using deep learning techniques. The system enables proactive maintenance 
    scheduling for Rolls-Royce Aerospace.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("🚀 Features")
        st.write("""
        - **Data Exploration**: Visualize engine degradation patterns
        - **Model Training**: Train LSTM/RNN models with custom configurations
        - **Predictions**: Generate RUL predictions for test engines
        - **Evaluation**: Comprehensive metrics and visualizations
        - **Multiple Datasets**: Support for FD001-FD004 datasets
        - **Model Artifacts**: Save and load trained models
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("📖 Quick Start")
        st.write("""
        1. **Data Exploration**: Start by exploring the dataset
        2. **Model Training**: Configure and train your model
        3. **Save Artifact**: Save your trained model
        4. **Predictions**: Generate RUL predictions
        5. **Evaluation**: Review metrics and performance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset information
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📁 Dataset Information")
    dataset_info = {
        "FD001": {"Engines": 100, "Conditions": 1, "Fault Modes": 1, "Description": "HPC Degradation"},
        "FD002": {"Engines": 260, "Conditions": 6, "Fault Modes": 1, "Description": "HPC Degradation"},
        "FD003": {"Engines": 100, "Conditions": 1, "Fault Modes": 2, "Description": "HPC + Fan Degradation"},
        "FD004": {"Engines": 248, "Conditions": 6, "Fault Modes": 2, "Description": "HPC + Fan Degradation"}
    }
    
    info = dataset_info[dataset]
    st.info(f"""
    **Current Dataset: {dataset}**
    - Training Engines: {info['Engines']}
    - Operating Conditions: {info['Conditions']}
    - Fault Modes: {info['Fault Modes']}
    - Description: {info['Description']}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_exploration(dataset, data_path):
    """Display data exploration page"""
    st.header("📊 Data Exploration")
    
    if st.button("Load Dataset", type="primary", use_container_width=True):
        with st.spinner("Loading data..."):
            try:
                preprocessor = TurbofanDataPreprocessor(data_path=data_path)
                train_data, test_data, test_rul = preprocessor.load_data(
                    train_file=f'train_{dataset}.txt',
                    test_file=f'test_{dataset}.txt',
                    rul_file=f'RUL_{dataset}.txt'
                )
                
                st.session_state.preprocessor = preprocessor
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.test_rul = test_rul
                
                st.success("✅ Data loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                return
    
    if st.session_state.train_data is not None:
        train_data = st.session_state.train_data
        test_data = st.session_state.test_data
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Engines", train_data['unit_id'].nunique())
        with col2:
            st.metric("Training Samples", len(train_data))
        with col3:
            st.metric("Test Engines", test_data['unit_id'].nunique())
        with col4:
            st.metric("Test Samples", len(test_data))
        
        st.markdown("---")
        
        # Data visualization with tabs
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Visualizations", "📋 Data Preview", "📊 Statistics"])
        
        with viz_tab1:
            st.subheader("Data Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Data Distribution", use_container_width=True):
                    with st.spinner("Generating visualization..."):
                        try:
                            preprocessor = st.session_state.preprocessor
                            preprocessor.visualize_data_distribution()
                            if os.path.exists('visualizations/data_distribution.png'):
                                st.image('visualizations/data_distribution.png', use_column_width=True)
                                st.success("✅ Visualization generated!")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("Generate Sensor Trends", use_container_width=True):
                    with st.spinner("Generating visualization..."):
                        try:
                            preprocessor = st.session_state.preprocessor
                            preprocessor.visualize_sensor_trends(n_engines=3, n_sensors=6)
                            if os.path.exists('visualizations/sensor_trends.png'):
                                st.image('visualizations/sensor_trends.png', use_column_width=True)
                                st.success("✅ Visualization generated!")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        with viz_tab2:
            st.subheader("Data Preview")
            data_tab1, data_tab2 = st.tabs(["Training Data", "Test Data"])
            
            with data_tab1:
                st.dataframe(train_data.head(100), use_container_width=True)
            
            with data_tab2:
                st.dataframe(test_data.head(100), use_container_width=True)
        
        with viz_tab3:
            st.subheader("Statistics")
            stat_tab1, stat_tab2 = st.tabs(["Training Statistics", "Test Statistics"])
            
            with stat_tab1:
                st.dataframe(train_data.describe(), use_container_width=True)
            
            with stat_tab2:
                st.dataframe(test_data.describe(), use_container_width=True)
        
        # RUL distribution
        st.markdown("---")
        st.subheader("RUL Distribution")
        if st.button("Calculate and Plot RUL", use_container_width=True):
            with st.spinner("Calculating RUL..."):
                try:
                    preprocessor = st.session_state.preprocessor
                    train_rul = preprocessor.calculate_rul(train_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.hist(train_rul, bins=50, edgecolor='#3a3a3a', alpha=0.8, color='#ffffff')
                    ax.set_xlabel('Remaining Useful Life (Cycles)', fontsize=12, color='#e0e0e0')
                    ax.set_ylabel('Frequency', fontsize=12, color='#e0e0e0')
                    ax.set_title('RUL Distribution in Training Data', fontsize=14, fontweight='bold', color='#e0e0e0')
                    ax.grid(True, alpha=0.3, linestyle='--', color='#3a3a3a')
                    ax.set_facecolor('#0a0a0a')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('#3a3a3a')
                    ax.spines['left'].set_color('#3a3a3a')
                    ax.tick_params(colors='#e0e0e0')
                    st.pyplot(fig)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean RUL", f"{train_rul.mean():.2f} cycles")
                    with col2:
                        st.metric("Median RUL", f"{np.median(train_rul):.2f} cycles")
                    with col3:
                        st.metric("Min RUL", f"{train_rul.min():.2f} cycles")
                    with col4:
                        st.metric("Max RUL", f"{train_rul.max():.2f} cycles")
                    
                except Exception as e:
                    st.error(f"Error calculating RUL: {e}")

def show_model_training(dataset, data_path, model_type, sequence_length, 
                       lstm_units, dropout_rate, learning_rate, epochs, batch_size):
    """Display model training page"""
    st.header("🤖 Model Training")
    
    # Check if data is loaded
    if st.session_state.train_data is None:
        st.warning("⚠️ Please load data in the Data Exploration tab first.")
        return
    
    # Model configuration summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Model Configuration")
        st.write(f"- **Model Type**: {model_type}")
        st.write(f"- **Sequence Length**: {sequence_length}")
        st.write(f"- **LSTM Units**: {lstm_units}")
        st.write(f"- **Dropout Rate**: {dropout_rate}")
        st.write(f"- **Learning Rate**: {learning_rate}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Training Configuration")
        st.write(f"- **Epochs**: {epochs}")
        st.write(f"- **Batch Size**: {batch_size}")
        st.write(f"- **Dataset**: {dataset}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Preparing data and training model..."):
            try:
                # Get preprocessor and data
                preprocessor = st.session_state.preprocessor
                train_data = st.session_state.train_data
                
                # Prepare training data
                X_train, y_train, feature_cols = preprocessor.prepare_train_data(use_sensors_only=True)
                X_test, y_test = preprocessor.prepare_test_data(feature_cols)
                
                # Scale features
                X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
                
                # Create sequences
                X_train_seq, y_train_seq = preprocessor.create_sequences(
                    X_train_scaled, y_train,
                    sequence_length=sequence_length,
                    step_size=1
                )
                
                # Train/validation split
                from sklearn.model_selection import train_test_split
                X_train_final, X_val, y_train_final, y_val = train_test_split(
                    X_train_seq, y_train_seq, test_size=0.2, random_state=42
                )
                
                # Build and compile model
                input_shape = (X_train_final.shape[1], X_train_final.shape[2])
                model = RULPredictor(
                    input_shape=input_shape,
                    model_type=model_type,
                    units=lstm_units,
                    dropout_rate=dropout_rate
                )
                
                model.compile_model(learning_rate=learning_rate)
                
                # Display model architecture
                with st.expander("📐 View Model Architecture"):
                    try:
                        import io
                        from contextlib import redirect_stdout
                        f = io.StringIO()
                        with redirect_stdout(f):
                            model.get_model_summary()
                        summary = f.getvalue()
                        st.code(summary)
                    except:
                        st.write(f"**Model Type**: {model_type}")
                        st.write(f"**Input Shape**: {input_shape}")
                        st.write(f"**LSTM Units**: {lstm_units}")
                
                # Training progress
                st.info("⏳ Training started. This may take several minutes. Please wait...")
                
                # Train model
                history = model.train(
                    X_train_final, y_train_final,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
                
                # Store model and history
                st.session_state.model = model
                st.session_state.training_history = history
                
                st.success("✅ Model training completed!")
                
                # Display training history
                st.subheader("📈 Training History")
                if history is not None:
                    plot_training_history(history)
                
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

def show_model_artifacts():
    """Display model artifacts management page"""
    st.header("📦 Model Artifacts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Save Current Model")
        if st.session_state.model is None:
            st.warning("⚠️ No model trained. Train a model first in the Model Training tab.")
        else:
            model_name = st.text_input("Model Name", value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            save_description = st.text_area("Description (optional)")
            
            if st.button("💾 Save Model Artifact", type="primary", use_container_width=True):
                try:
                    # Get model configuration
                    config = {
                        'dataset': st.sidebar.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"]) if 'dataset' in st.session_state else 'FD001',
                        'model_type': st.session_state.model.model_type,
                        'input_shape': list(st.session_state.model.input_shape),
                        'units': st.session_state.model.units,
                        'dropout_rate': st.session_state.model.dropout_rate,
                        'sequence_length': 50,  # Default, should be stored
                        'description': save_description,
                        'created': datetime.now().isoformat()
                    }
                    
                    # Save artifact
                    artifact_dir = save_model_artifact(
                        st.session_state.model,
                        st.session_state.preprocessor,
                        config,
                        metrics=st.session_state.predictions.get('metrics') if st.session_state.predictions else None,
                        history=st.session_state.training_history
                    )
                    
                    st.success(f"✅ Model saved to {artifact_dir}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error saving model: {e}")
    
    with col2:
        st.subheader("📂 Saved Models")
        artifacts = list_model_artifacts()
        
        if artifacts:
            for artifact in artifacts:
                with st.expander(f"📦 {artifact['name']}"):
                    config = artifact['config']
                    st.write(f"**Dataset**: {config.get('dataset', 'Unknown')}")
                    st.write(f"**Model Type**: {config.get('model_type', 'Unknown')}")
                    st.write(f"**LSTM Units**: {config.get('units', 'Unknown')}")
                    st.write(f"**Created**: {datetime.fromtimestamp(artifact['created']).strftime('%Y-%m-%d %H:%M:%S')}")
                    if config.get('description'):
                        st.write(f"**Description**: {config['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📥 Load", key=f"load_{artifact['name']}"):
                            try:
                                model, config, scaler = load_model_artifact(artifact['path'])
                                st.session_state.model = model
                                st.session_state.loaded_model_path = artifact['path']
                                if st.session_state.preprocessor:
                                    st.session_state.preprocessor.scaler = scaler
                                st.success(f"✅ Model loaded!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{artifact['name']}"):
                            try:
                                shutil.rmtree(artifact['path'])
                                st.success("✅ Model deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
        else:
            st.info("No saved models found. Train and save a model to see it here.")

def show_predictions():
    """Display predictions page"""
    st.header("📈 RUL Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Training tab or load a model from Model Artifacts.")
        return
    
    if st.session_state.test_data is None:
        st.warning("⚠️ Please load data first in the Data Exploration tab.")
        return
    
    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            try:
                model = st.session_state.model
                preprocessor = st.session_state.preprocessor
                test_data = st.session_state.test_data
                test_rul = st.session_state.test_rul
                
                # Get feature columns
                feature_cols = [col for col in test_data.columns if col.startswith('sensor_')]
                
                # Prepare test sequences
                sequence_length = model.input_shape[0]
                X_test_sequences = []
                y_test_sequences = []
                
                unique_engines = sorted(test_data['unit_id'].unique())
                
                progress_bar = st.progress(0)
                for idx, unit_id in enumerate(unique_engines):
                    unit_data = test_data[test_data['unit_id'] == unit_id]
                    unit_features = unit_data[feature_cols].values
                    unit_features_scaled = preprocessor.scaler.transform(unit_features)
                    
                    if len(unit_features_scaled) >= sequence_length:
                        X_test_sequences.append(unit_features_scaled[-sequence_length:])
                    else:
                        padded = np.zeros((sequence_length, unit_features_scaled.shape[1]))
                        padded[-len(unit_features_scaled):] = unit_features_scaled
                        X_test_sequences.append(padded)
                    
                    unit_rul = test_rul.iloc[idx]['RUL']
                    y_test_sequences.append(unit_rul)
                    progress_bar.progress((idx + 1) / len(unique_engines))
                
                X_test_seq = np.array(X_test_sequences)
                y_test_seq = np.array(y_test_sequences)
                
                # Make predictions
                y_pred = model.predict(X_test_seq)
                y_pred = y_pred.flatten()
                
                # Calculate metrics
                evaluator = RULEvaluator()
                metrics = evaluator.calculate_metrics(y_test_seq, y_pred)
                
                # Store predictions
                st.session_state.predictions = {
                    'y_true': y_test_seq,
                    'y_pred': y_pred,
                    'engine_ids': unique_engines,
                    'metrics': metrics
                }
                
                st.success("✅ Predictions generated successfully!")
                
                # Display predictions
                st.subheader("📊 Prediction Results")
                
                # Create DataFrame
                results_df = pd.DataFrame({
                    'Engine ID': unique_engines,
                    'Actual RUL': y_test_seq,
                    'Predicted RUL': y_pred,
                    'Error': y_test_seq - y_pred,
                    'Absolute Error': np.abs(y_test_seq - y_pred)
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"rul_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                st.subheader("📈 Prediction Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    evaluator.plot_predictions_vs_actual(y_test_seq, y_pred)
                    if os.path.exists('visualizations/predictions_vs_actual.png'):
                        st.image('visualizations/predictions_vs_actual.png', use_column_width=True)
                
                with col2:
                    evaluator.plot_error_distribution(y_test_seq, y_pred)
                    if os.path.exists('visualizations/error_distribution.png'):
                        st.image('visualizations/error_distribution.png', use_column_width=True)
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

def show_evaluation():
    """Display evaluation page"""
    st.header("📉 Model Evaluation")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please generate predictions first in the Predictions tab.")
        return
    
    predictions = st.session_state.predictions
    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    
    # Calculate metrics
    evaluator = RULEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    # Display metrics with gradient cards
    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error")
    with col3:
        st.metric("R² Score", f"{metrics['R2_Score']:.4f}", help="Coefficient of Determination")
    with col4:
        st.metric("MSE", f"{metrics['MSE']:.2f}", help="Mean Squared Error")
    with col5:
        st.metric("PHM08 Score", f"{metrics['PHM08_Score']:.2f}", help="PHM08 Challenge Score")
    
    st.markdown("---")
    
    # Performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Error Statistics")
        errors = y_true - y_pred
        st.write(f"- **Mean Error**: {errors.mean():.2f} cycles")
        st.write(f"- **Median Error**: {np.median(errors):.2f} cycles")
        st.write(f"- **Std Error**: {errors.std():.2f} cycles")
        st.write(f"- **Min Error**: {errors.min():.2f} cycles")
        st.write(f"- **Max Error**: {errors.max():.2f} cycles")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Absolute Error Statistics")
        absolute_errors = np.abs(errors)
        st.write(f"- **Mean Absolute Error**: {absolute_errors.mean():.2f} cycles")
        st.write(f"- **Median Absolute Error**: {np.median(absolute_errors):.2f} cycles")
        st.write(f"- **Std Absolute Error**: {absolute_errors.std():.2f} cycles")
        st.write(f"- **Min Absolute Error**: {absolute_errors.min():.2f} cycles")
        st.write(f"- **Max Absolute Error**: {absolute_errors.max():.2f} cycles")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training history
    if st.session_state.training_history is not None:
        st.markdown("---")
        st.subheader("📈 Training History")
        plot_training_history(st.session_state.training_history)

def plot_training_history(history):
    """Plot training history with black and white theme styling"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Black and white theme color scheme
    train_color = '#ffffff'  # White
    val_color = '#808080'    # Gray
    bg_color = '#0a0a0a'     # Black background
    text_color = '#e0e0e0'   # Light text
    grid_color = '#3a3a3a'   # Dark gray grid
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', marker='o', color=train_color, linewidth=2, markersize=4)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', marker='s', color=val_color, linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, color=text_color)
    axes[0].set_ylabel('Loss', fontsize=12, color=text_color)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold', color=text_color)
    axes[0].legend(fontsize=10, framealpha=0.9, facecolor='#1a1a1a', edgecolor=grid_color, labelcolor=text_color)
    axes[0].grid(True, alpha=0.3, linestyle='--', color=grid_color)
    axes[0].set_facecolor(bg_color)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_color(grid_color)
    axes[0].spines['left'].set_color(grid_color)
    axes[0].tick_params(colors=text_color)
    
    # MAE plot
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Training MAE', marker='o', color=train_color, linewidth=2, markersize=4)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', marker='s', color=val_color, linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, color=text_color)
        axes[1].set_ylabel('MAE', fontsize=12, color=text_color)
        axes[1].set_title('Mean Absolute Error Over Epochs', fontsize=14, fontweight='bold', color=text_color)
        axes[1].legend(fontsize=10, framealpha=0.9, facecolor='#1a1a1a', edgecolor=grid_color, labelcolor=text_color)
        axes[1].grid(True, alpha=0.3, linestyle='--', color=grid_color)
        axes[1].set_facecolor(bg_color)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['bottom'].set_color(grid_color)
        axes[1].spines['left'].set_color(grid_color)
        axes[1].tick_params(colors=text_color)
    
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
