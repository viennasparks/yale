#!/usr/bin/env python3
"""
Main entry point for the Aptamer Discovery Platform.

This script coordinates the entire workflow for aptamer discovery, including data loading,
feature extraction, model training, and aptamer selection.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import warnings

from src.utils.logger import setup_logging
from src.utils.validators import validate_config
from src.data_processing.data_loader import AptamerDataLoader
from src.data_processing.preprocessor import AptamerPreprocessor
from src.feature_extraction.sequence_features import SequenceFeatureExtractor
from src.feature_extraction.structure_prediction import StructurePredictor
from src.models.binding_affinity import BindingAffinityPredictor
from src.models.cross_reactivity import CrossReactivityAnalyzer
from src.aptamer_selection.selector import AptamerSelector
from src.aptamer_selection.specificity_optimizer import AptamerOptimizer
from src.visualization.plot_utils import AptamerVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Aptamer Discovery Platform')
    
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data file (CSV format)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['fentanyl', 'methamphetamine', 'benzodiazepine', 'xylazine', 'nitazene'],
                        help='Target molecules to focus on')
    parser.add_argument('--num-aptamers', type=int, default=5,
                        help='Number of aptamers to select per target')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform aptamer optimization for improved specificity')
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable visualization generation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError("Invalid configuration")
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        sys.exit(1)

def setup_environment(args, config):
    """
    Set up the environment for the platform.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    config : dict
        Configuration dictionary
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.info("CUDA is available. Using GPU for neural network models.")
        else:
            logger.info("CUDA not available. Using CPU for neural network models.")
    except ImportError:
        logger.warning("PyTorch not found. Neural network models will not be available.")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.configure(handlers=[
        {"sink": sys.stderr, "level": log_level},
        {"sink": os.path.join(args.output, "aptamer_discovery.log"), "level": log_level}
    ])
    
    # Filter warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    logger.info(f"Environment set up with random seed: {args.seed}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Target molecules: {args.targets}")

def run_data_processing(data_path, config):
    """
    Load and process data.
    
    Parameters
    ----------
    data_path : str
        Path to input data
    config : dict
        Configuration dictionary
    
    Returns
    -------
    pd.DataFrame
        Processed data
    """
    logger.info("=== Data Processing Phase ===")
    
    # Load data
    data_loader = AptamerDataLoader()
    logger.info(f"Loading data from {data_path}")
    df = data_loader.load_from_csv(data_path)
    logger.info(f"Loaded {len(df)} aptamer records")
    
    # Preprocess data
    preprocessor = AptamerPreprocessor(config.get('data_processing', {}))
    df = preprocessor.clean_data(df)
    logger.info(f"After cleaning: {len(df)} aptamer records")
    
    # Normalize target names
    if 'Target_Name' in df.columns:
        df = preprocessor.normalize_target_names(df)
        logger.info(f"Normalized target names. Found {df['Target_Name'].nunique()} unique targets")
    
    return df

def extract_features(df, config):
    """
    Extract features from aptamer sequences.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aptamer data
    config : dict
        Configuration dictionary
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features
    """
    logger.info("=== Feature Extraction Phase ===")
    
    # Determine sequence column
    seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
    if seq_col not in df.columns:
        logger.error("No sequence column found in data")
        sys.exit(1)
    
    # Extract sequence features
    logger.info("Extracting sequence features")
    feature_extractor = SequenceFeatureExtractor(config.get('feature_extraction', {}))
    sequence_features = feature_extractor.extract_all_features(df[seq_col].tolist())
    
    # Extract structural features
    logger.info("Predicting structures and extracting structural features")
    structure_predictor = StructurePredictor(config.get('structure_prediction', {}))
    structure_features = structure_predictor.predict_and_analyze_structures(df[seq_col].tolist())
    
    # Combine features
    logger.info("Combining all features")
    columns_to_exclude = ['sequence'] if 'sequence' in sequence_features.columns else []
    combined_df = pd.concat([
        df.reset_index(drop=True),
        sequence_features.reset_index(drop=True).drop(columns=columns_to_exclude),
        structure_features.reset_index(drop=True).drop(columns=['sequence'])
    ], axis=1)
    
    logger.info(f"Generated {len(combined_df.columns) - len(df.columns)} new features")
    
    return combined_df

def train_models(df, config):
    """
    Train binding affinity and cross-reactivity models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aptamer data and features
    config : dict
        Configuration dictionary
    
    Returns
    -------
    tuple
        (binding_model, cross_reactivity_model)
    """
    logger.info("=== Model Training Phase ===")
    
    # Split data
    data_loader = AptamerDataLoader()
    train_df, val_df, test_df = data_loader.split_data(
        df,
        test_size=config.get('data_processing', {}).get('test_size', 0.2),
        validation_size=config.get('data_processing', {}).get('validation_split', 0.1),
        random_state=42
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Train binding affinity model
    binding_model = BindingAffinityPredictor(config.get('modeling', {}).get('binding_affinity', {}))
    
    # Check if we have binding data
    binding_col = None
    for col in df.columns:
        if 'binding' in col.lower() or 'affinity' in col.lower() or 'kd' in col.lower():
            binding_col = col
            break
    
    if binding_col:
        logger.info(f"Training binding affinity model using column: {binding_col}")
        binding_metrics = binding_model.train(train_df, binding_col, validation_df=val_df)
        logger.info(f"Binding model training metrics: {binding_metrics}")
    else:
        logger.warning("No binding affinity data found. Using default model for predictions.")
    
    # Train cross-reactivity model if target information is available
    cross_reactivity_model = None
    if 'Target_Name' in df.columns:
        logger.info("Training cross-reactivity model")
        cross_reactivity_model = CrossReactivityAnalyzer(config.get('modeling', {}).get('cross_reactivity', {}))
        cross_metrics = cross_reactivity_model.train_cross_reactivity_model(train_df, validation_df=val_df)
        logger.info(f"Cross-reactivity model training metrics: {cross_metrics}")
    else:
        logger.warning("No target information. Cross-reactivity analysis will be limited.")
    
    # Evaluate models on test set
    if binding_col and binding_model.model is not None:
        test_metrics = binding_model.evaluate_model(test_df, binding_col)
        logger.info(f"Binding model test metrics: {test_metrics}")
    
    return binding_model, cross_reactivity_model

def select_aptamers(df, targets, binding_model, cross_reactivity_model, config, num_aptamers=5):
    """
    Select optimal aptamers for each target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aptamer data and features
    targets : list
        List of target molecules
    binding_model : BindingAffinityPredictor
        Trained binding affinity model
    cross_reactivity_model : CrossReactivityAnalyzer
        Trained cross-reactivity model
    config : dict
        Configuration dictionary
    num_aptamers : int, optional
        Number of aptamers to select per target, by default 5
    
    Returns
    -------
    pd.DataFrame
        DataFrame with selected aptamers
    """
    logger.info("=== Aptamer Selection Phase ===")
    
    # Create aptamer selector
    selector = AptamerSelector(config.get('aptamer_selection', {}))
    
    # Set models
    selector.binding_model = binding_model
    selector.cross_reactivity_model = cross_reactivity_model
    
    # Select optimal aptamers
    logger.info(f"Selecting {num_aptamers} aptamers for each of {len(targets)} targets")
    selected_aptamers = selector.select_optimal_aptamers(df, targets, n_per_target=num_aptamers)
    
    # Verify quality
    verified_aptamers = selector.verify_aptamer_quality(selected_aptamers, targets)
    
    logger.info(f"Selected {len(verified_aptamers)} aptamers in total")
    
    return verified_aptamers

def optimize_aptamers(selected_df, targets, binding_model, cross_reactivity_model, config):
    """
    Optimize selected aptamers for improved specificity.
    
    Parameters
    ----------
    selected_df : pd.DataFrame
        DataFrame with selected aptamers
    targets : list
        List of target molecules
    binding_model : BindingAffinityPredictor
        Trained binding affinity model
    cross_reactivity_model : CrossReactivityAnalyzer
        Trained cross-reactivity model
    config : dict
        Configuration dictionary
    
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized aptamers
    """
    logger.info("=== Aptamer Optimization Phase ===")
    
    # Create aptamer optimizer
    optimizer = AptamerOptimizer(config.get('aptamer_selection', {}))
    
    # Set models
    optimizer.binding_model = binding_model
    optimizer.cross_reactivity_model = cross_reactivity_model
    
    # Run optimization in parallel for each target
    optimized_aptamers = optimizer.run_parallel_optimization(
        selected_df, 
        targets=targets,
        num_optimized=3  # Fewer optimized aptamers as they take more resources
    )
    
    logger.info(f"Optimized {len(optimized_aptamers)} aptamers")
    
    return optimized_aptamers

def generate_visualizations(df, selected_df, optimized_df, binding_model, cross_reactivity_model, output_dir):
    """
    Generate visualizations for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with aptamer data and features
    selected_df : pd.DataFrame
        DataFrame with selected aptamers
    optimized_df : pd.DataFrame
        DataFrame with optimized aptamers (or None if optimization was not performed)
    binding_model : BindingAffinityPredictor
        Trained binding affinity model
    cross_reactivity_model : CrossReactivityAnalyzer
        Trained cross-reactivity model
    output_dir : str
        Output directory
    """
    logger.info("=== Visualization Phase ===")
    
    # Create visualizer
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    visualizer = AptamerVisualizer({'output_directory': vis_dir})
    
    # Generate various plots
    logger.info("Generating distribution plots")
    visualizer.plot_sequence_length_distribution(df, by_target=True)
    visualizer.plot_gc_content_distribution(df, by_target=True)
    
    # Plot feature importance if available
    if binding_model and hasattr(binding_model, 'model') and hasattr(binding_model, 'feature_names'):
        logger.info("Generating feature importance plot")
        visualizer.plot_feature_importance(binding_model.model, binding_model.feature_names)
    
    # Plot cross-reactivity matrix if available
    if cross_reactivity_model and len(selected_df) > 0 and 'Target_Name' in selected_df.columns:
        logger.info("Generating cross-reactivity matrix")
        visualizer.plot_cross_reactivity_matrix(selected_df)
    
    # Plot selected aptamers
    if len(selected_df) > 0:
        logger.info("Generating selected aptamers dashboard")
        visualizer.plot_binding_vs_specificity(selected_df)
        visualizer.create_dashboard(selected_df)
        
        # Plot structure for a few selected aptamers
        logger.info("Generating structure visualizations")
        seq_col = 'Sequence' if 'Sequence' in selected_df.columns else 'sequence'
        struct_col = 'predicted_structure'
        
        if seq_col in selected_df.columns and struct_col in selected_df.columns:
            for i in range(min(3, len(selected_df))):
                seq = selected_df.iloc[i][seq_col]
                struct = selected_df.iloc[i][struct_col]
                target = selected_df.iloc[i]['Target_Name'] if 'Target_Name' in selected_df.columns else f"Target {i+1}"
                
                visualizer.plot_structure_visualization(
                    seq, struct, 
                    title=f"Aptamer Structure for {target}",
                    output_path=os.path.join(vis_dir, f"structure_{i+1}.png")
                )
    
    # Plot optimized aptamers
    if optimized_df is not None and len(optimized_df) > 0:
        logger.info("Generating optimized aptamers visualizations")
        visualizer.plot_binding_vs_specificity(
            optimized_df, 
            output_path=os.path.join(vis_dir, "optimized_binding_vs_specificity.png")
        )
        
        # Compare before/after optimization
        if len(selected_df) > 0 and 'specificity_score' in selected_df.columns and 'specificity_score' in optimized_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate average specificity by target
            if 'Target_Name' in selected_df.columns and 'Target_Name' in optimized_df.columns:
                selected_spec = selected_df.groupby('Target_Name')['specificity_score'].mean()
                optimized_spec = optimized_df.groupby('Target_Name')['specificity_score'].mean()
                
                targets = sorted(set(selected_spec.index) | set(optimized_spec.index))
                selected_values = [selected_spec.get(t, 0) for t in targets]
                optimized_values = [optimized_spec.get(t, 0) for t in targets]
                
                x = np.arange(len(targets))
                bar_width = 0.35
                
                ax.bar(x - bar_width/2, selected_values, bar_width, label='Original')
                ax.bar(x + bar_width/2, optimized_values, bar_width, label='Optimized')
                
                ax.set_xlabel('Target')
                ax.set_ylabel('Average Specificity Score')
                ax.set_title('Specificity Improvement After Optimization')
                ax.set_xticks(x)
                ax.set_xticklabels(targets, rotation=45, ha='right')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, "optimization_comparison.png"), dpi=300)
                plt.close()

def save_results(selected_df, optimized_df, binding_model, cross_reactivity_model, output_dir):
    """
    Save results to output directory.
    
    Parameters
    ----------
    selected_df : pd.DataFrame
        DataFrame with selected aptamers
    optimized_df : pd.DataFrame
        DataFrame with optimized aptamers (or None if optimization was not performed)
    binding_model : BindingAffinityPredictor
        Trained binding affinity model
    cross_reactivity_model : CrossReactivityAnalyzer
        Trained cross-reactivity model
    output_dir : str
        Output directory
    """
    logger.info("=== Saving Results ===")
    
    # Create directory for CSV files
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save selected aptamers
    if len(selected_df) > 0:
        selected_path = os.path.join(csv_dir, 'selected_aptamers.csv')
        selected_df.to_csv(selected_path, index=False)
        logger.info(f"Selected aptamers saved to {selected_path}")
    
    # Save optimized aptamers
    if optimized_df is not None and len(optimized_df) > 0:
        optimized_path = os.path.join(csv_dir, 'optimized_aptamers.csv')
        optimized_df.to_csv(optimized_path, index=False)
        logger.info(f"Optimized aptamers saved to {optimized_path}")
    
    # Create directory for models
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save models
    if binding_model and hasattr(binding_model, 'model'):
        binding_path = os.path.join(model_dir, 'binding_affinity_model.pkl')
        binding_model.save_model(binding_path)
        logger.info(f"Binding affinity model saved to {binding_path}")
    
    if cross_reactivity_model and hasattr(cross_reactivity_model, 'model'):
        crossreact_path = os.path.join(model_dir, 'cross_reactivity_model.pkl')
        cross_reactivity_model.save_model(crossreact_path)
        logger.info(f"Cross-reactivity model saved to {crossreact_path}")
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Aptamer Discovery Platform: Summary Report ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Selected Aptamers ===\n")
        if len(selected_df) > 0:
            f.write(f"Total selected aptamers: {len(selected_df)}\n")
            if 'Target_Name' in selected_df.columns:
                for target, group in selected_df.groupby('Target_Name'):
                    f.write(f"  - {target}: {len(group)} aptamers\n")
        else:
            f.write("No aptamers were selected.\n")
        
        f.write("\n=== Optimized Aptamers ===\n")
        if optimized_df is not None and len(optimized_df) > 0:
            f.write(f"Total optimized aptamers: {len(optimized_df)}\n")
            if 'Target_Name' in optimized_df.columns:
                for target, group in optimized_df.groupby('Target_Name'):
                    f.write(f"  - {target}: {len(group)} aptamers\n")
        else:
            f.write("No aptamers were optimized.\n")
        
        f.write("\n=== Files Generated ===\n")
        f.write(f"- Selected aptamers: csv/selected_aptamers.csv\n")
        if optimized_df is not None and len(optimized_df) > 0:
            f.write(f"- Optimized aptamers: csv/optimized_aptamers.csv\n")
        f.write(f"- Visualizations: visualizations/\n")
        f.write(f"- Models: models/\n")
    
    logger.info(f"Summary report saved to {summary_path}")

def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Print banner
    print("=" * 80)
    print("                  APTAMER DISCOVERY PLATFORM")
    print("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    setup_environment(args, config)
    
    # Start timing
    start_time = datetime.now()
    logger.info(f"Starting aptamer discovery at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Process data
        df = run_data_processing(args.data, config)
        
        # Extract features
        df_with_features = extract_features(df, config)
        
        # Train models
        binding_model, cross_reactivity_model = train_models(df_with_features, config)
        
        # Select aptamers
        selected_aptamers = select_aptamers(
            df_with_features, args.targets, 
            binding_model, cross_reactivity_model,
            config, args.num_aptamers
        )
        
        # Optimize aptamers if requested
        optimized_aptamers = None
        if args.optimize:
            optimized_aptamers = optimize_aptamers(
                selected_aptamers, args.targets,
                binding_model, cross_reactivity_model,
                config
            )
        
        # Generate visualizations
        if not args.no_vis:
            import matplotlib.pyplot as plt
            generate_visualizations(
                df_with_features, selected_aptamers, optimized_aptamers,
                binding_model, cross_reactivity_model,
                args.output
            )
        
        # Save results
        save_results(
            selected_aptamers, optimized_aptamers,
            binding_model, cross_reactivity_model,
            args.output
        )
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed = end_time - start_time
        logger.info(f"Aptamer discovery completed in {elapsed}")
        
    except Exception as e:
        logger.exception(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
