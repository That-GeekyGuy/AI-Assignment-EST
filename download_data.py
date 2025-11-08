"""
Utility script to help download or verify the NASA Turbofan Engine Dataset
"""

import os
import urllib.request
import zipfile
import shutil

def download_dataset_kaggle():
    """
    Instructions for downloading from Kaggle
    """
    print("="*70)
    print("NASA Turbofan Engine Dataset Verification")
    print("="*70)
    print()
    print("The dataset should be in the 'CMAPSSData/' directory with the following structure:")
    print("   CMAPSSData/")
    print("   ├── train_FD001.txt")
    print("   ├── test_FD001.txt")
    print("   ├── RUL_FD001.txt")
    print("   ├── train_FD002.txt")
    print("   ├── test_FD002.txt")
    print("   ├── RUL_FD002.txt")
    print("   ├── train_FD003.txt")
    print("   ├── test_FD003.txt")
    print("   ├── RUL_FD003.txt")
    print("   ├── train_FD004.txt")
    print("   ├── test_FD004.txt")
    print("   └── RUL_FD004.txt")
    print()
    print("If the dataset is not present, you can download it from:")
    print("1. Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("2. GitHub: https://github.com/Lucky-Loek/CMAPSS_data")
    print("3. NASA Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print()

def verify_data(data_path='CMAPSSData'):
    """
    Verify that the required data files exist
    
    Args:
        data_path: Path to data directory
    """
    # Check for all 4 datasets
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    required_files = []
    for dataset in datasets:
        required_files.extend([f'train_{dataset}.txt', f'test_{dataset}.txt', f'RUL_{dataset}.txt'])
    
    print("="*70)
    print("Verifying Data Files")
    print("="*70)
    print()
    
    if not os.path.exists(data_path):
        print(f"❌ Data directory '{data_path}' does not exist!")
        print(f"   Please create the directory and add the dataset files.")
        return False
    
    all_exists = True
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file} found ({file_size:.2f} MB)")
        else:
            print(f"❌ {file} not found!")
            all_exists = False
    
    print()
    if all_exists:
        print("✅ All required data files are present!")
        print(f"✓ Found all 4 datasets: FD001, FD002, FD003, FD004")
        return True
    else:
        missing_count = sum(1 for f in required_files if not os.path.exists(os.path.join(data_path, f)))
        print(f"❌ {missing_count} data file(s) are missing.")
        print("   Please ensure the CMAPSSData folder contains all dataset files.")
        print()
        download_dataset_kaggle()
        return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_data()
    else:
        download_dataset_kaggle()
        verify_data()


