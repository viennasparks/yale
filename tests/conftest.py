"""
Configuration for pytest tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / 'data'

@pytest.fixture
def sample_sequences():
    """Sample aptamer sequences for testing."""
    return [
        'ACGTACGTACGTACGTACGT',
        'GCTAGCTAGCTAGCTAGCTA',
        'CGCGCGCGCGCGCGCGCGCG',
        'ATATATATATATATATATATATATAT',
        'GGGAGGGAGGGATTTCCCGAGGGA'
    ]

@pytest.fixture
def sample_structures():
    """Sample secondary structures in dot-bracket notation."""
    return [
        '((((.....))))........',
        '(((.((....)).)))....',
        '((((((.....))))))...',
        '((((....))))...((((....))))',
        '(((((....)))))..((((....))))'
    ]

@pytest.fixture
def sample_aptamer_df():
    """Sample DataFrame with aptamer data."""
    return pd.DataFrame({
        'Sequence_ID': ['apt1', 'apt2', 'apt3', 'apt4', 'apt5'],
        'Sequence': [
            'ACGTACGTACGTACGTACGT',
            'GCTAGCTAGCTAGCTAGCTA',
            'CGCGCGCGCGCGCGCGCGCG',
            'ATATATATATATATATATATATATAT',
            'GGGAGGGAGGGATTTCCCGAGGGA'
        ],
        'Target_Name': ['fentanyl', 'methamphetamine', 'benzodiazepine', 'xylazine', 'nitazene'],
        'length': [20, 20, 20, 24, 24],
        'GC_Content': [50.0, 50.0, 100.0, 0.0, 66.7],
        'binding_affinity': [0.8, 0.7, 0.9, 0.6, 0.85]
    })

@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        'data_processing': {
            'test_size': 0.2,
            'validation_split': 0.1,
            'random_state': 42
        },
        'feature_extraction': {
            'k_mer_sizes': [1, 2, 3],
            'include_structural_features': True,
            'include_thermodynamic_features': True
        },
        'structure_prediction': {
            'algorithm': 'mfe',
            'temperature': 37.0
        },
        'modeling': {
            'binding_affinity': {
                'model_type': 'xgboost',
                'n_estimators': 100
            },
            'cross_reactivity': {
                'model_type': 'xgboost',
                'n_estimators': 100
            }
        },
        'aptamer_selection': {
            'specificity_weight': 0.6,
            'binding_affinity_weight': 0.4
        },
        'targets': [
            {'name': 'fentanyl', 'smiles': 'CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3'},
            {'name': 'methamphetamine', 'smiles': 'CC(NC)CC1=CC=CC=C1'},
            {'name': 'benzodiazepine', 'smiles': 'CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3'},
            {'name': 'xylazine', 'smiles': 'CN(C)C(=NC1=CC=C(C=C1)C)NC2=CC=CC=C2'},
            {'name': 'nitazene', 'smiles': 'CCC(=NCCC)N1CCN(CC1)C2=C3C=C(C=CC3=NC=C2)C(=O)OC'}
        ]
    }
