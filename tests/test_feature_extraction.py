"""
Tests for the feature extraction module.
"""
import pytest
import numpy as np
from src.feature_extraction.sequence_features import SequenceFeatureExtractor
from src.feature_extraction.structure_prediction import StructurePredictor

def test_sequence_feature_extractor(sample_sequences):
    """Test extracting features from sequences."""
    extractor = SequenceFeatureExtractor()
    features_df = extractor.extract_all_features(sample_sequences)
    
    # Check that we have a row for each sequence
    assert len(features_df) == len(sample_sequences)
    
    # Check that we have basic features
    assert 'gc_content' in features_df.columns
    assert 'length' in features_df.columns
    
    # Check GC content calculation
    assert features_df['gc_content'].iloc[0] == 50.0  # ACGTACGTACGTACGTACGT
    assert features_df['gc_content'].iloc[2] == 100.0  # CGCGCGCGCGCGCGCGCGCG
    
    # Check sequence length
    assert features_df['length'].iloc[0] == 20
    assert features_df['length'].iloc[3] == 24

def test_kmer_feature_extraction(sample_sequences):
    """Test k-mer feature extraction."""
    extractor = SequenceFeatureExtractor()
    kmer_features = extractor.extract_kmer_features(sample_sequences, k_values=[1, 2])
    
    # Check that we have features for both k=1 and k=2
    assert 1 in kmer_features
    assert 2 in kmer_features
    
    # Check dimensions
    assert kmer_features[1].shape[0] == len(sample_sequences)
    assert kmer_features[1].shape[1] == 4  # A, C, G, T
    assert kmer_features[2].shape[1] == 16  # All dinucleotide combinations

def test_structure_predictor(sample_sequences):
    """Test structure prediction."""
    predictor = StructurePredictor()
    
    # Test MFE prediction
    for seq in sample_sequences:
        structure, energy = predictor.predict_mfe_structure(seq)
        
        # Check that we have a valid structure
        assert isinstance(structure, str)
        assert len(structure) == len(seq)
        assert all(c in '.()' for c in structure)
        
        # Check that the energy is negative (stable)
        assert energy <= 0

def test_ensemble_properties(sample_sequences):
    """Test calculating ensemble properties."""
    predictor = StructurePredictor()
    
    for seq in sample_sequences:
        props = predictor.predict_ensemble_properties(seq)
        
        # Check that we have the expected properties
        assert 'ensemble_free_energy' in props
        assert 'ensemble_diversity' in props
        assert 'centroid_structure' in props
        
        # Check that centroid structure is valid
        centroid = props['centroid_structure']
        assert isinstance(centroid, str)
        assert len(centroid) == len(seq)
        assert all(c in '.()' for c in centroid)

def test_structural_features(sample_structures):
    """Test calculating structural features from dot-bracket notation."""
    predictor = StructurePredictor()
    
    for structure in sample_structures:
        features = predictor.calculate_structural_features(structure)
        
        # Check that we have the expected features
        assert 'unpaired_percentage' in features
        assert 'paired_percentage' in features
        assert 'stem_count' in features
        
        # Check that percentages add up to approximately 100%
        assert abs(features['unpaired_percentage'] + features['paired_percentage'] - 100) < 1e-6
        
        # Check that stem count is reasonable
        assert features['stem_count'] >= 0
        assert features['stem_count'] <= len(structure) // 2

def test_batch_structure_prediction(sample_sequences):
    """Test batch prediction of structures."""
    predictor = StructurePredictor()
    
    results_df = predictor.predict_and_analyze_structures(sample_sequences)
    
    # Check that we have a row for each sequence
    assert len(results_df) == len(sample_sequences)
    
    # Check that we have the expected columns
    assert 'sequence' in results_df.columns
    assert 'predicted_structure' in results_df.columns
    assert 'energy' in results_df.columns
    assert 'unpaired_percentage' in results_df.columns
