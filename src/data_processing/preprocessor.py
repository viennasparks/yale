import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import re
from loguru import logger
from src.utils.validators import validate_nucleotide_sequence

class AptamerPreprocessor:
    """
    Class for preprocessing and cleaning aptamer data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AptamerPreprocessor.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        self.target_mappings = self._initialize_target_mappings()
        logger.info("Initialized AptamerPreprocessor")
        
    def _initialize_target_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings for target name normalization.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping raw target names to normalized names
        """
        # Define mappings for target name normalization
        base_mappings = {
            "Fentanyl": "FENTANYL",
            "fentanyl": "FENTANYL",
            "Acetyl Fentanyl": "ACETYL_FENTANYL",
            "acetyl fentanyl": "ACETYL_FENTANYL",
            "Acetyl-Fentanyl": "ACETYL_FENTANYL",
            "Furanyl Fentanyl": "FURANYL_FENTANYL",
            "furanyl fentanyl": "FURANYL_FENTANYL",
            "Furanyl-Fentanyl": "FURANYL_FENTANYL",
            "METHAMPHETAMINE": "METHAMPHETAMINE",
            "Methamphetamine": "METHAMPHETAMINE",
            "methamphetamine": "METHAMPHETAMINE",
            "BENZODIAZEPINE": "BENZODIAZEPINE",
            "Benzodiazepine": "BENZODIAZEPINE",
            "benzodiazepine": "BENZODIAZEPINE",
            "XYLAZINE": "XYLAZINE",
            "Xylazine": "XYLAZINE",
            "xylazine": "XYLAZINE",
            "NITAZENE": "NITAZENE",
            "Nitazene": "NITAZENE",
            "nitazene": "NITAZENE"
        }
        
        # Add specific benzodiazepine types
        benzos = ["Diazepam", "Alprazolam", "Clonazepam", "Lorazepam", "Temazepam", "Midazolam"]
        for benzo in benzos:
            base_mappings[benzo.upper()] = "BENZODIAZEPINE"
            base_mappings[benzo.lower()] = "BENZODIAZEPINE"
            base_mappings[benzo] = "BENZODIAZEPINE"
        
        # Add specific nitazene types
        nitazenes = ["Isotonitazene", "Metonitazene", "Etonitazene", "Protonitazene"]
        for nitazene in nitazenes:
            base_mappings[nitazene.upper()] = "NITAZENE"
            base_mappings[nitazene.lower()] = "NITAZENE"
            base_mappings[nitazene] = "NITAZENE"
        
        # Add custom mappings from config if available
        if self.config and 'target_mappings' in self.config:
            base_mappings.update(self.config['target_mappings'])
            
        return base_mappings
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the aptamer data by removing duplicates, handling missing values,
        and standardizing sequence representation.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        logger.info("Cleaning aptamer data")
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Remove rows with missing sequences
        if "Sequence" in cleaned_df.columns:
            missing_sequence = cleaned_df["Sequence"].isna()
            if missing_sequence.any():
                missing_count = missing_sequence.sum()
                logger.warning(f"Removing {missing_count} rows with missing sequences")
                cleaned_df = cleaned_df.dropna(subset=["Sequence"])
        
        # Standardize sequences to uppercase
        if "Sequence" in cleaned_df.columns:
            logger.debug("Standardizing sequences to uppercase")
            cleaned_df["Sequence"] = cleaned_df["Sequence"].str.upper()
        
        # Calculate GC content if missing
        if "Sequence" in cleaned_df.columns:
            if "GC_Content" not in cleaned_df.columns:
                logger.debug("Calculating GC content")
                cleaned_df["GC_Content"] = cleaned_df["Sequence"].apply(self._calculate_gc_content)
            elif cleaned_df["GC_Content"].isna().any():
                logger.debug("Filling missing GC content values")
                cleaned_df["GC_Content"] = cleaned_df["GC_Content"].fillna(
                    cleaned_df["Sequence"].apply(self._calculate_gc_content)
                )
        
        # Calculate sequence length if missing
        if "Sequence" in cleaned_df.columns and "length" not in cleaned_df.columns:
            logger.debug("Calculating sequence lengths")
            cleaned_df["length"] = cleaned_df["Sequence"].str.len()
        
        # Remove sequences with non-standard nucleotides or invalid characters
        if "Sequence" in cleaned_df.columns:
            valid_indices = cleaned_df["Sequence"].apply(validate_nucleotide_sequence)
            invalid_count = (~valid_indices).sum()
            
            if invalid_count > 0:
                logger.warning(f"Removing {invalid_count} sequences with non-standard nucleotides")
                cleaned_df = cleaned_df[valid_indices].reset_index(drop=True)
        
        # Remove duplicate sequences
        if "Sequence" in cleaned_df.columns:
            dup_count = cleaned_df.duplicated(subset=["Sequence"]).sum()
            if dup_count > 0:
                logger.debug(f"Removing {dup_count} duplicate sequences")
                cleaned_df = cleaned_df.drop_duplicates(subset=["Sequence"])
        
        # Reset index after dropping rows
        cleaned_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Data cleaning complete. Original: {len(df)} rows, Cleaned: {len(cleaned_df)} rows")
        return cleaned_df
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """
        Calculate the GC content of a sequence.
        
        Parameters
        ----------
        sequence : str
            Nucleotide sequence
            
        Returns
        -------
        float
            GC content as a percentage
        """
        if not sequence:
            return 0.0
        
        sequence = str(sequence).upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_length = len(sequence)
        
        if total_length == 0:
            return 0.0
        
        return (gc_count / total_length) * 100
    
    def normalize_target_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize target names for consistency.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with normalized target names
        """
        logger.info("Normalizing target names")
        
        normalized_df = df.copy()
        
        # Apply the mapping to normalize target names
        if "Target_Name" in normalized_df.columns:
            # Count unique targets before normalization
            unique_before = normalized_df["Target_Name"].nunique()
            original_targets = normalized_df["Target_Name"].unique()
            
            normalized_df["Target_Name"] = normalized_df["Target_Name"].apply(
                lambda x: self.target_mappings.get(x, x)
            )
            
            # Count unique targets after normalization
            unique_after = normalized_df["Target_Name"].nunique()
            normalized_targets = normalized_df["Target_Name"].unique()
            
            logger.debug(f"Target normalization: {unique_before} unique targets -> {unique_after} normalized targets")
            logger.debug(f"Original targets: {original_targets}")
            logger.debug(f"Normalized targets: {normalized_targets}")
        else:
            logger.warning("No 'Target_Name' column found for normalization")
        
        return normalized_df
    
    def filter_by_target(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Filter the DataFrame to include only rows with a specific target.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        target : str
            Target name to filter by
            
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        if "Target_Name" not in df.columns:
            logger.error("DataFrame does not contain a 'Target_Name' column")
            raise ValueError("DataFrame does not contain a 'Target_Name' column")
        
        # Normalize the target name for consistency
        normalized_target = target.upper().replace(" ", "_").replace("-", "_")
        logger.info(f"Filtering data for target: {normalized_target}")
        
        # Apply the filter
        filtered_df = df[df["Target_Name"].str.upper().str.replace(" ", "_").str.replace("-", "_") == normalized_target]
        
        if filtered_df.empty:
            logger.warning(f"No data found for target '{target}' (normalized: '{normalized_target}')")
        else:
            logger.debug(f"Found {len(filtered_df)} rows for target '{target}'")
        
        return filtered_df
    
    def handle_imbalanced_targets(self, df: pd.DataFrame, 
                                strategy: str = 'undersample', 
                                random_state: int = 42) -> pd.DataFrame:
        """
        Handle imbalanced target classes using undersampling or oversampling.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        strategy : str, optional
            Balancing strategy ('undersample', 'oversample', or 'smote'), by default 'undersample'
        random_state : int, optional
            Random seed, by default 42
            
        Returns
        -------
        pd.DataFrame
            Balanced DataFrame
        """
        if 'Target_Name' not in df.columns:
            logger.error("Missing 'Target_Name' column for handling imbalanced targets")
            raise ValueError("DataFrame must contain a 'Target_Name' column")
        
        target_counts = df['Target_Name'].value_counts()
        logger.info(f"Target distribution before balancing: {dict(target_counts)}")
        
        if target_counts.nunique() == 1:
            logger.info("Data already balanced - all targets have the same count")
            return df
        
        if strategy == 'undersample':
            # Undersample majority classes to match the minority class
            min_class_count = target_counts.min()
            balanced_dfs = []
            
            for target in target_counts.index:
                target_df = df[df['Target_Name'] == target]
                
                if len(target_df) > min_class_count:
                    # Undersample this target
                    sampled = target_df.sample(min_class_count, random_state=random_state)
                    balanced_dfs.append(sampled)
                else:
                    # Keep all rows for this target
                    balanced_dfs.append(target_df)
            
            result_df = pd.concat(balanced_dfs, ignore_index=True)
            logger.info(f"Undersampled to {min_class_count} samples per target")
            
        elif strategy == 'oversample':
            # Oversample minority classes to match the majority class
            max_class_count = target_counts.max()
            balanced_dfs = []
            
            for target in target_counts.index:
                target_df = df[df['Target_Name'] == target]
                
                if len(target_df) < max_class_count:
                    # Oversample this target with replacement
                    sampled = target_df.sample(max_class_count, replace=True, random_state=random_state)
                    balanced_dfs.append(sampled)
                else:
                    # Keep all rows for this target
                    balanced_dfs.append(target_df)
            
            result_df = pd.concat(balanced_dfs, ignore_index=True)
            logger.info(f"Oversampled to {max_class_count} samples per target")
            
        elif strategy == 'smote':
            # Use SMOTE for oversampling (if imblearn is installed)
            try:
                from imblearn.over_sampling import SMOTE
                from sklearn.preprocessing import LabelEncoder
                
                # Prepare features and target
                X = df.drop('Target_Name', axis=1)
                y = df['Target_Name']
                
                # Encode categorical features
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                # Apply SMOTE
                smote = SMOTE(random_state=random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Reconstruct DataFrame
                result_df = X_resampled.copy()
                result_df['Target_Name'] = y_resampled
                
                logger.info(f"Applied SMOTE oversampling, resulting in {len(result_df)} samples")
                
            except ImportError:
                logger.warning("SMOTE requires the 'imblearn' package. Falling back to regular oversampling.")
                return self.handle_imbalanced_targets(df, strategy='oversample', random_state=random_state)
        else:
            logger.warning(f"Unknown balancing strategy: '{strategy}'. No balancing applied.")
            return df
        
        logger.debug(f"Target distribution after balancing: {dict(result_df['Target_Name'].value_counts())}")
        return result_df
