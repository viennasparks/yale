import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import os
from loguru import logger
from src.models.binding_affinity import BindingAffinityPredictor
from src.models.cross_reactivity import CrossReactivityAnalyzer
from src.feature_extraction.structure_prediction import StructurePredictor
import random
from tqdm import tqdm
import multiprocessing
import concurrent.futures

class AptamerSelector:
    """
    Select optimal aptamers based on binding affinity and cross-reactivity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AptamerSelector.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        
        # Selection parameters
        self.min_binding_score = self.config.get('min_binding_score', 0.8)
        self.max_cross_reactivity = self.config.get('max_cross_reactivity', 0.2)
        self.specificity_weight = self.config.get('specificity_weight', 0.6)
        self.binding_affinity_weight = self.config.get('binding_affinity_weight', 0.4)
        self.structural_stability_weight = self.config.get('structural_stability_weight', 0.3)
        
        # Initialize models
        self.binding_model = None
        self.cross_reactivity_model = None
        self.structure_predictor = StructurePredictor(self.config.get('structure_prediction', {}))
        
        logger.info(f"Initialized AptamerSelector with specificity_weight={self.specificity_weight}, "
                   f"binding_affinity_weight={self.binding_affinity_weight}")
    
    def select_optimal_aptamers(self, df: pd.DataFrame, targets: List[str], 
                              n_per_target: int = 5) -> pd.DataFrame:
        """
        Select optimal aptamers for each target with minimal cross-reactivity.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        targets : List[str]
            List of target molecules
        n_per_target : int, optional
            Number of aptamers to select per target, by default 5
            
        Returns
        -------
        pd.DataFrame
            DataFrame with selected aptamers
        """
        logger.info(f"Selecting {n_per_target} optimal aptamers for each of {len(targets)} targets")
        
        # Initialize models if not already initialized
        if self.binding_model is None:
            self.binding_model = BindingAffinityPredictor(self.config.get('binding_affinity', {}))
        
        if self.cross_reactivity_model is None:
            self.cross_reactivity_model = CrossReactivityAnalyzer(self.config.get('cross_reactivity', {}))
        
        # Train cross-reactivity model if df has target information
        if 'Target_Name' in df.columns:
            logger.debug("Training cross-reactivity model")
            self.cross_reactivity_model.train_cross_reactivity_model(df)
            
            # Predict cross-reactivity
            df_with_cross_reactivity = self.cross_reactivity_model.predict_cross_reactivity(df)
            df_with_specificity = self.cross_reactivity_model.calculate_specificity_score(df_with_cross_reactivity)
        else:
            logger.warning("No 'Target_Name' column in DataFrame. Cannot evaluate cross-reactivity.")
            df_with_specificity = df.copy()
            # Add dummy columns for specificity
            df_with_specificity['specificity_score'] = 1.0
            df_with_specificity['entropy_specificity'] = 1.0
            df_with_specificity['combined_specificity'] = 1.0
        
        # Train binding affinity model if df has binding data
        binding_column = None
        for col in df.columns:
            if 'binding' in col.lower() or 'affinity' in col.lower() or 'kd' in col.lower():
                binding_column = col
                break
        
        if binding_column:
            logger.debug(f"Training binding affinity model using column: {binding_column}")
            self.binding_model.train(df, binding_column)
        
        # Predict binding affinity if not already present
        if 'predicted_affinity' not in df_with_specificity.columns:
            try:
                df_with_specificity['predicted_affinity'] = self.binding_model.predict(df_with_specificity)
            except RuntimeError:
                logger.warning("Binding affinity model not trained and no binding data available.")
                # Assign random scores for demonstration
                df_with_specificity['predicted_affinity'] = np.random.uniform(0.5, 1.0, size=len(df_with_specificity))
        
        # Predict structure and calculate stability for each sequence
        logger.debug("Predicting structures and calculating stability")
        sequences = df_with_specificity['Sequence'].tolist() if 'Sequence' in df_with_specificity.columns else df_with_specificity['sequence'].tolist()
        
        # Using parallel processing for structure prediction
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, multiprocessing.cpu_count())) as executor:
            structure_futures = [executor.submit(self._calculate_stability_score, seq) for seq in sequences]
            
            # Collect results
            stability_scores = []
            for future in tqdm(concurrent.futures.as_completed(structure_futures), 
                              total=len(sequences), 
                              desc="Predicting structures"):
                stability_scores.append(future.result())
        
        df_with_specificity['stability_score'] = stability_scores
        
        # Calculate combined score for selection
        logger.debug("Calculating combined selection score")
        df_with_specificity = self._calculate_selection_score(df_with_specificity)
        
        # Select top aptamers for each target
        selected_aptamers = []
        
        for target in targets:
            logger.debug(f"Selecting aptamers for target: {target}")
            
            # Filter by target if target information is available
            if 'Target_Name' in df_with_specificity.columns:
                target_df = df_with_specificity[df_with_specificity['Target_Name'].str.contains(target, case=False)]
                
                if target_df.empty:
                    logger.warning(f"No aptamers found for target: {target}")
                    continue
            else:
                # If no target information, use all aptamers (for demonstration)
                target_df = df_with_specificity.copy()
            
            # Select top N by selection score
            top_aptamers = target_df.sort_values('selection_score', ascending=False).head(n_per_target)
            
            # Add target information if not already present
            if 'Target_Name' not in top_aptamers.columns:
                top_aptamers['Target_Name'] = target
            
            selected_aptamers.append(top_aptamers)
        
        # Combine results
        if selected_aptamers:
            result = pd.concat(selected_aptamers, ignore_index=True)
            logger.info(f"Selected {len(result)} aptamers for {len(targets)} targets")
            return result
        else:
            logger.warning("No aptamers were selected")
            return pd.DataFrame()
    
    def _calculate_stability_score(self, sequence: str) -> float:
        """
        Calculate structural stability score for an aptamer sequence.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        float
            Stability score (higher is more stable)
        """
        try:
            # Predict MFE structure and energy
            structure, mfe = self.structure_predictor.predict_mfe_structure(sequence)
            
            # Get ensemble properties
            ensemble_props = self.structure_predictor.predict_ensemble_properties(sequence)
            
            # Calculate structural features
            struct_features = self.structure_predictor.calculate_structural_features(structure)
            
            # Factors that contribute to stability:
            # 1. More negative MFE
            # 2. Lower ensemble diversity (more well-defined structure)
            # 3. Higher percentage of paired bases
            # 4. More stems
            
            mfe_contribution = min(-mfe / 10.0, 1.0)  # Normalize to [0, 1]
            
            diversity = ensemble_props.get("ensemble_diversity", 0)
            diversity_contribution = 1.0 - min(diversity / len(sequence), 1.0)  # Lower is better
            
            paired_percent = struct_features.get("paired_percentage", 0)
            paired_contribution = paired_percent / 100.0
            
            stem_count = struct_features.get("stem_count", 0)
            stem_contribution = min(stem_count / 5.0, 1.0)  # Normalize to [0, 1]
            
            # Overall stability score
            stability_score = (
                0.4 * mfe_contribution + 
                0.3 * diversity_contribution + 
                0.2 * paired_contribution + 
                0.1 * stem_contribution
            )
            
            return stability_score
        except Exception as e:
            logger.warning(f"Error calculating stability score: {str(e)}")
            return 0.5  # Default score
    
    def _calculate_selection_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a combined selection score based on binding affinity, 
        cross-reactivity, and structural stability.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aptamer data including affinity and specificity scores
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added selection score
        """
        result_df = df.copy()
        
        # Normalize predicted_affinity to [0, 1] if needed
        if 'predicted_affinity' in result_df.columns:
            min_aff = result_df['predicted_affinity'].min()
            max_aff = result_df['predicted_affinity'].max()
            
            if max_aff > min_aff:
                result_df['affinity_normalized'] = (result_df['predicted_affinity'] - min_aff) / (max_aff - min_aff)
            else:
                result_df['affinity_normalized'] = 0.5  # Default if all values are the same
        else:
            result_df['affinity_normalized'] = 0.5  # Default score
        
        # Combine scores with weights
        result_df['selection_score'] = (
            self.binding_affinity_weight * result_df['affinity_normalized'] +
            self.specificity_weight * result_df['combined_specificity'] +
            self.structural_stability_weight * result_df['stability_score']
        ) / (self.binding_affinity_weight + self.specificity_weight + self.structural_stability_weight)
        
        return result_df
    
    def verify_aptamer_quality(self, df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
        """
        Verify the quality of selected aptamers by calculating additional metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with selected aptamers
        targets : List[str]
            List of target molecules
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional quality metrics
        """
        result_df = df.copy()
        
        # Calculate G-quadruplex potential
        if 'Sequence' in result_df.columns:
            sequences = result_df['Sequence'].tolist()
        else:
            sequences = result_df['sequence'].tolist()
        
        g4_scores = self.structure_predictor.get_g_quadruplex_propensity(sequences)
        result_df['g4_potential'] = g4_scores
        
        # Check for sequence motifs that might cause problems
        result_df['homopolymer_risk'] = [self._check_homopolymer_risk(seq) for seq in sequences]
        
        # Calculate Tm estimates
        thermo_results = self.structure_predictor.calculate_thermodynamic_stability(sequences)
        result_df['estimated_tm'] = [result['approximated_tm'] for result in thermo_results]
        result_df['stability_score_refined'] = [result['stability_score'] for result in thermo_results]
        
        # Add quality summary
        def get_quality_category(row):
            score = row['selection_score']
            if score >= 0.8:
                return 'Excellent'
            elif score >= 0.6:
                return 'Good'
            elif score >= 0.4:
                return 'Fair'
            else:
                return 'Poor'
        
        result_df['quality_category'] = result_df.apply(get_quality_category, axis=1)
        
        return result_df
    
    def _check_homopolymer_risk(self, sequence: str) -> float:
        """
        Check for homopolymer runs in a sequence, which can cause non-specific binding.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        float
            Risk score (higher means higher risk)
        """
        sequence = sequence.upper()
        risk_score = 0.0
        
        # Check for runs of 4+ of the same nucleotide
        for nt in 'ATGC':
            nt_run = nt * 4
            if nt_run in sequence:
                risk_score += 0.25
            
            nt_run = nt * 5
            if nt_run in sequence:
                risk_score += 0.25
                
            nt_run = nt * 6
            if nt_run in sequence:
                risk_score += 0.5
        
        # Check for alternating patterns which can be problematic
        for pattern in ['ATATAT', 'GCGCGC', 'TGTGTG', 'CACACA']:
            if pattern in sequence:
                risk_score += 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def export_selected_aptamers(self, df: pd.DataFrame, output_path: str, 
                               include_structures: bool = True) -> None:
        """
        Export selected aptamers to a CSV file with detailed information.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with selected aptamers
        output_path : str
            Path to save the CSV file
        include_structures : bool, optional
            Whether to include predicted structures, by default True
        """
        result_df = df.copy()
        
        # Add predicted structures if requested
        if include_structures and 'predicted_structure' not in result_df.columns:
            if 'Sequence' in result_df.columns:
                sequences = result_df['Sequence'].tolist()
            else:
                sequences = result_df['sequence'].tolist()
            
            # Predict structures
            structures = []
            for seq in sequences:
                structure, energy = self.structure_predictor.predict_mfe_structure(seq)
                structures.append(structure)
            
            result_df['predicted_structure'] = structures
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        result_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(result_df)} selected aptamers to {output_path}")
    
    def save_models(self, directory: str) -> None:
        """
        Save trained models to the specified directory.
        
        Parameters
        ----------
        directory : str
            Directory to save the models
        """
        os.makedirs(directory, exist_ok=True)
        
        if self.binding_model and hasattr(self.binding_model, 'model') and self.binding_model.model is not None:
            binding_path = os.path.join(directory, 'binding_affinity_model.pkl')
            self.binding_model.save_model(binding_path)
            logger.info(f"Binding affinity model saved to {binding_path}")
        
        if self.cross_reactivity_model and hasattr(self.cross_reactivity_model, 'model') and self.cross_reactivity_model.model is not None:
            crossreact_path = os.path.join(directory, 'cross_reactivity_model.pkl')
            self.cross_reactivity_model.save_model(crossreact_path)
            logger.info(f"Cross-reactivity model saved to {crossreact_path}")
    
    def load_models(self, directory: str) -> None:
        """
        Load trained models from the specified directory.
        
        Parameters
        ----------
        directory : str
            Directory containing the saved models
        """
        binding_path = os.path.join(directory, 'binding_affinity_model.pkl')
        if os.path.exists(binding_path):
            if self.binding_model is None:
                self.binding_model = BindingAffinityPredictor(self.config.get('binding_affinity', {}))
            self.binding_model.load_model(binding_path)
            logger.info(f"Binding affinity model loaded from {binding_path}")
        
        crossreact_path = os.path.join(directory, 'cross_reactivity_model.pkl')
        if os.path.exists(crossreact_path):
            if self.cross_reactivity_model is None:
                self.cross_reactivity_model = CrossReactivityAnalyzer(self.config.get('cross_reactivity', {}))
            self.cross_reactivity_model.load_model(crossreact_path)
            logger.info(f"Cross-reactivity model loaded from {crossreact_path}")
