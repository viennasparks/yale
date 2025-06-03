"""
Tests for the machine learning models.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from src.models.binding_affinity import BindingAffinityPredictor
from src.models.cross_reactivity import CrossReactivityAnalyzer

def create_synthetic_binding_data(n_samples=100, n_features=20):
    """Create synthetic data for binding affinity prediction."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
    
    # Scale target to [0, 1] range
    y = (y - y.min()) / (y.max() - y.min())
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['binding_affinity'] = y
    df['Sequence'] = [f'ACGT{i:03d}' for i in range(n_samples)]
    
    return df

def create_synthetic_crossreact_data(n_samples=100, n_features=20, n_targets=3):
    """Create synthetic data for cross-reactivity analysis."""
    X, _ = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
    
    # Create target labels
    targets = [f'target_{i}' for i in range(n_targets)]
    y = np.random.choice(targets, size=n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Name'] = y
    df['Sequence'] = [f'ACGT{i:03d}' for i in range(n_samples)]
    
    return df

def test_binding_affinity_model():
    """Test binding affinity prediction model."""
    # Create synthetic data
    df = create_synthetic_binding_data(n_samples=100)
    
    # Split data
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]
    
    # Initialize and train model
    predictor = BindingAffinityPredictor({'model_type': 'random_forest', 'n_estimators': 10})
    metrics = predictor.train(train_df, 'binding_affinity')
    
    # Check that the model was trained
    assert predictor.model is not None
    
    # Check metrics
    assert 'training_mse' in metrics
    assert 'training_r2' in metrics
    
    # Test predictions
    predictions = predictor.predict(test_df)
    assert len(predictions) == len(test_df)
    assert all(0 <= p <= 1 for p in predictions)
    
    # Test getting top candidates
    top_df = predictor.get_top_candidates(test_df, n=3)
    assert len(top_df) == 3
    assert 'predicted_affinity' in top_df.columns

def test_cross_reactivity_model():
    """Test cross-reactivity prediction model."""
    # Create synthetic data
    df = create_synthetic_crossreact_data(n_samples=100, n_targets=3)
    
    # Split data
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]
    
    # Initialize and train model
    analyzer = CrossReactivityAnalyzer({'model_type': 'random_forest', 'n_estimators': 10})
    metrics = analyzer.train_cross_reactivity_model(train_df)
    
    # Check that the model was trained
    assert analyzer.model is not None
    
    # Check metrics
    assert 'training_accuracy' in metrics
    assert 'training_confusion_matrix' in metrics
    
    # Test predictions
    pred_df = analyzer.predict_cross_reactivity(test_df)
    assert len(pred_df) == len(test_df)
    assert 'predicted_target' in pred_df.columns
    
    # Test cross-reactivity identification
    crossreact_df = analyzer.identify_cross_reactive_aptamers(pred_df)
    assert 'is_cross_reactive' in crossreact_df.columns
    assert 'specificity_score' in crossreact_df.columns

def test_model_save_load(tmp_path):
    """Test saving and loading models."""
    # Create synthetic data
    df = create_synthetic_binding_data(n_samples=50)
    
    # Train a model
    predictor = BindingAffinityPredictor({'model_type': 'random_forest', 'n_estimators': 10})
    predictor.train(df, 'binding_affinity')
    
    # Save the model
    save_path = tmp_path / "test_model.pkl"
    predictor.save_model(save_path)
    
    # Load the model in a new instance
    new_predictor = BindingAffinityPredictor()
    new_predictor.load_model(save_path)
    
    # Test that the loaded model works
    predictions = new_predictor.predict(df)
    assert len(predictions) == len(df)
