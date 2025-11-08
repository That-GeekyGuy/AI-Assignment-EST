"""
Quick test script to verify data loading works with CMAPSSData folder
"""

from data_preprocessing import TurbofanDataPreprocessor
import os

def test_data_loading():
    """Test loading data from CMAPSSData folder"""
    
    print("="*70)
    print("Testing Data Loading from CMAPSSData Folder")
    print("="*70)
    print()
    
    # Test FD001 dataset
    print("Testing FD001 dataset...")
    preprocessor = TurbofanDataPreprocessor(data_path='CMAPSSData')
    
    try:
        train_data, test_data, test_rul = preprocessor.load_data(
            train_file='train_FD001.txt',
            test_file='test_FD001.txt',
            rul_file='RUL_FD001.txt'
        )
        
        print(f"✓ Successfully loaded FD001 dataset")
        print(f"  Training data shape: {train_data.shape}")
        print(f"  Test data shape: {test_data.shape}")
        print(f"  Test RUL shape: {test_rul.shape}")
        print(f"  Training columns: {list(train_data.columns[:5])}...")
        print(f"  Number of engines in training: {train_data['unit_id'].nunique()}")
        print(f"  Number of engines in test: {test_data['unit_id'].nunique()}")
        print()
        
        # Test other datasets
        for dataset in ['FD002', 'FD003', 'FD004']:
            print(f"Testing {dataset} dataset...")
            preprocessor = TurbofanDataPreprocessor(data_path='CMAPSSData')
            train_data, test_data, test_rul = preprocessor.load_data(
                train_file=f'train_{dataset}.txt',
                test_file=f'test_{dataset}.txt',
                rul_file=f'RUL_{dataset}.txt'
            )
            print(f"  ✓ {dataset} loaded: Train={train_data.shape}, Test={test_data.shape}, RUL={test_rul.shape}")
            print(f"    Engines: Train={train_data['unit_id'].nunique()}, Test={test_data['unit_id'].nunique()}")
            print()
        
        print("="*70)
        print("✅ All datasets loaded successfully!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_data_loading()




