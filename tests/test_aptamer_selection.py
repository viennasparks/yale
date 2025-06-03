"""
Tests for the aptamer selection module.
"""
import pytest
import pandas as pd
import numpy as np
from src.aptamer_selection.selector import AptamerSelector
from src.aptamer_selection.specificity_optimizer import AptamerOptimizer

def create_test_aptamer_df():
    """Create a test DataFrame for aptamer selection."""
    return pd.DataFrame({
        'Sequence': [
            'ACGTACGTACGTACGTACGT',
            'GCTAGCTAGCTAGCTAGCTA',
            'CGCGCGCGCGCGCGCGCGCG',
            'ATATATATATATATATATATATATAT',
            'GGGAGGGAGGGATTTCCCGAGGGA',
            'ACGTACGTATTTACGTACGT',
            'GCTAGCTAGAAAAGCTAGCTA',
            'CGCGCGCGCTTTCGCGCGCG',
            'ATATATATATCCCATATATATATATAT',
            'GGGAGGGAGGGTTTCCCGAGGGA'
        ],
        'Target_Name': [
            'fentanyl', 'fentanyl', 'methamphetamine', 'methamphetamine', 'benzodiazepine',
            'benzodiazepine', 'xylazine', 'xylazine', 'nitazene', 'nitazene'
        ],
        'predicted_affinity': [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.95, 0.65, 0.77, 0.88],
        'specificity_score': [0.9, 0.85, 0.7, 0.95, 0.8, 0.75, 0.6, 0.9, 0.7, 0.85],
        'stability_score': [0.7, 0.65, 0.8, 0.75, 0.9, 0.6, 0.85, 0.7, 0.75, 0.8],
        'length': [20, 20, 20, 24, 24, 20, 20, 20, 28, 24],
        'gc_content': [50.0, 50.0, 100.0, 0.0, 66.7, 45.0, 45.0, 75.0, 14.3, 66.7],
        'combined_specificity': [0.85, 0.82, 0.75, 0.92, 0.78, 0.72, 0.65, 0.88, 0.72, 0.80],
        'predicted_structure': [
            '(((...)))..........', '.(((...))).........', '((((....))))........',
            '.(((...))).............', '(((((...)))))(((....)))',
            '(((...)))..((...)).', '.(((...)))..((..))', '(((((...)))))((..))',
            '.(((...)))................', '(((...)))...(((....)))'
        ]
    })

def test_aptamer_selector():
    """Test the AptamerSelector class."""
    # Create test data
    df = create_test_aptamer_df()
    
    # Initialize selector
    selector = AptamerSelector({
        'specificity_weight': 0.6,
        'binding_affinity_weight': 0.4,
        'structural_stability_weight': 0.2
    })
    
    # Select aptamers
    targets = ['fentanyl', 'methamphetamine', 'benzodiazepine', 'xylazine', 'nitazene']
    selected = selector.select_optimal_aptamers(df, targets, n_per_target=1)
    
    # Check results
    assert len(selected) == 5  # One for each target
    assert set(selected['Target_Name']) == set(targets)
    assert 'selection_score' in selected.columns

def test_selector_quality_verification():
    """Test the aptamer quality verification."""
    # Create test data
    df = create_test_aptamer_df()
    
    # Initialize selector
    selector = AptamerSelector()
    
    # Select aptamers
    targets = ['fentanyl', 'methamphetamine']
    selected = selector.select_optimal_aptamers(df, targets, n_per_target=1)
    
    # Verify quality
    verified = selector.verify_aptamer_quality(selected, targets)
    
    # Check results
    assert len(verified) == len(selected)
    assert 'g4_potential' in verified.columns
    assert 'homopolymer_risk' in verified.columns

def test_aptamer_optimizer():
    """Test the AptamerOptimizer class."""
    # Create test data
    df = create_test_aptamer_df()
    
    # Initialize optimizer with mocked models
    optimizer = AptamerOptimizer({
        'optimization_iterations': 5,  # Small number for testing
        'population_size': 10,
        'mutation_rate': 0.1
    })
    
    # Mock the evaluation function to return random scores
    def mock_evaluate(population, target, non_targets):
        return [np.random.random() for _ in population]
    
    optimizer._evaluate_population = mock_evaluate
    
    # Run optimization
    target = 'fentanyl'
    non_targets = ['methamphetamine', 'benzodiazepine', 'xylazine', 'nitazene']
    
    # Filter data for target
    target_df = df[df['Target_Name'] == target]
    
    optimized = optimizer.optimize_aptamers(target_df, target, non_targets, num_optimized=2)
    
    # Check results
    assert len(optimized) == 2
    assert 'optimization_fitness' in optimized.columns
    assert all(optimized['Target_Name'] == target)
