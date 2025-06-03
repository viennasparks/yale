"""
Tests for the data processing module.
"""
import pytest
import pandas as pd
import os
import tempfile
from src.data_processing.data_loader import AptamerDataLoader
from src.data_processing.preprocessor import AptamerPreprocessor

def test_data_loader_csv(sample_aptamer_df):
    """Test loading data from CSV file."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        sample_aptamer_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        # Test loading the CSV
        loader = AptamerDataLoader()
        df = loader.load_from_csv(tmp_path)
        
        # Verify the data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_aptamer_df)
        assert 'Sequence' in df.columns
        assert 'Target_Name' in df.columns
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_data_validation(sample_aptamer_df):
    """Test data validation."""
    loader = AptamerDataLoader(validation=True)
    
    # Create invalid sequence
    invalid_df = sample_aptamer_df.copy()
    invalid_df.loc[0, 'Sequence'] = 'ACGT123XYZ'
    
    # Create a temporary CSV file with invalid data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        invalid_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        # Test validation failure
        with pytest.raises(ValueError):
            loader.load_from_csv(tmp_path)
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_preprocessor_clean_data(sample_aptamer_df):
    """Test data cleaning."""
    preprocessor = AptamerPreprocessor()
    
    # Add some duplicates and mixed cases
    dirty_df = pd.concat([sample_aptamer_df, sample_aptamer_df.iloc[[0, 1]]], ignore_index=True)
    dirty_df.loc[0, 'Sequence'] = dirty_df.loc[0, 'Sequence'].lower()
    
    # Clean the data
    clean_df = preprocessor.clean_data(dirty_df)
    
    # Check that duplicates were removed
    assert len(clean_df) == len(sample_aptamer_df)
    
    # Check that sequences were standardized to uppercase
    assert clean_df['Sequence'].iloc[0] == sample_aptamer_df['Sequence'].iloc[0].upper()

def test_target_name_normalization():
    """Test normalizing target names."""
    preprocessor = AptamerPreprocessor()
    
    # Create a DataFrame with various target name formats
    df = pd.DataFrame({
        'Sequence': ['ACGT', 'GCTA', 'CGCG', 'ATAT'],
        'Target_Name': ['fentanyl', 'Fentanyl', 'FENTANYL', 'Methamphetamine']
    })
    
    # Normalize target names
    normalized_df = preprocessor.normalize_target_names(df)
    
    # Check that all fentanyl variants are normalized
    fentanyl_rows = normalized_df[normalized_df['Target_Name'] == 'FENTANYL']
    assert len(fentanyl_rows) == 3
    
    # Check that methamphetamine is normalized
    assert normalized_df['Target_Name'].iloc[3] == 'METHAMPHETAMINE'

def test_filter_by_target(sample_aptamer_df):
    """Test filtering by target."""
    preprocessor = AptamerPreprocessor()
    
    # Filter for fentanyl
    fentanyl_df = preprocessor.filter_by_target(sample_aptamer_df, 'fentanyl')
    assert len(fentanyl_df) == 1
    assert fentanyl_df['Target_Name'].iloc[0] == 'fentanyl'
    
    # Filter for non-existent target
    empty_df = preprocessor.filter_by_target(sample_aptamer_df, 'nonexistent')
    assert len(empty_df) == 0
