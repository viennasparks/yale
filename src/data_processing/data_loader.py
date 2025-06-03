import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from Bio import SeqIO
from loguru import logger

from src.utils.validators import validate_nucleotide_sequence

class AptamerDataLoader:
    """
    Loads and processes aptamer data from various file formats.
    """
    
    def __init__(self, validation: bool = True):
        """
        Initialize the AptamerDataLoader.
        
        Parameters
        ----------
        validation : bool, optional
            Whether to validate the data after loading, by default True
        """
        self.validation = validation
        logger.info("Initialized AptamerDataLoader")
        
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load aptamer data from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing aptamer data
        """
        logger.info(f"Loading aptamer data from CSV file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {len(df)} rows from {file_path}")
            
            if self.validation:
                self._validate_aptamer_data(df)
                
            return df
        except Exception as e:
            logger.error(f"Failed to load aptamer data from {file_path}: {str(e)}")
            raise IOError(f"Failed to load aptamer data from {file_path}: {str(e)}")
    
    def load_from_fasta(self, file_path: str) -> pd.DataFrame:
        """
        Load aptamer sequences from a FASTA file.
        
        Parameters
        ----------
        file_path : str
            Path to the FASTA file
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing aptamer sequences
        """
        logger.info(f"Loading aptamer data from FASTA file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            sequences = []
            
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append({
                    "Sequence_ID": record.id,
                    "Sequence": str(record.seq),
                    "Description": record.description
                })
            
            df = pd.DataFrame(sequences)
            logger.debug(f"Loaded {len(df)} sequences from {file_path}")
            
            if self.validation:
                self._validate_aptamer_data(df)
            
            return df
        except Exception as e:
            logger.error(f"Failed to load FASTA data from {file_path}: {str(e)}")
            raise IOError(f"Failed to load FASTA data from {file_path}: {str(e)}")
    
    def _validate_aptamer_data(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded aptamer data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
            
        Raises
        ------
        ValueError
            If the data is not in the expected format
        """
        logger.debug("Validating aptamer data")
        
        # Check if DataFrame is empty
        if df.empty:
            logger.error("Empty DataFrame")
            raise ValueError("DataFrame is empty")
        
        # Check if required column exists
        if "Sequence" not in df.columns:
            logger.error("Missing required column: Sequence")
            raise ValueError("DataFrame must contain a 'Sequence' column")
        
        # Check if sequence column contains valid DNA/RNA sequences
        invalid_sequences = df["Sequence"].apply(
            lambda seq: not validate_nucleotide_sequence(seq)
        )
        
        if invalid_sequences.any():
            invalid_indices = df.index[invalid_sequences].tolist()
            logger.warning(f"Invalid nucleotide sequences detected at indices: {invalid_indices}")
            
            # Extract and log a few examples of invalid sequences
            if len(invalid_indices) > 0:
                examples = df.loc[invalid_indices[:3], "Sequence"].tolist()
                logger.warning(f"Examples of invalid sequences: {examples}")
            
            raise ValueError(f"Invalid nucleotide sequences detected at {len(invalid_indices)} positions")
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  validation_size: float = 0.1, random_state: int = 42, 
                  stratify: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training, validation, and testing sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        test_size : float, optional
            Fraction of data to use for testing, by default 0.2
        validation_size : float, optional
            Fraction of data to use for validation, by default 0.1
        random_state : int, optional
            Random seed for reproducibility, by default 42
        stratify : Optional[str], optional
            Column to use for stratified splitting, by default None
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, validation_df, test_df) - Training, validation, and testing DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Splitting data (test: {test_size}, validation: {validation_size})")
        
        # Extract stratify column if specified
        stratify_col = None
        if stratify and stratify in df.columns:
            stratify_col = df[stratify]
            logger.debug(f"Using stratified splitting based on column: {stratify}")
        
        # First split: separate the test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
        
        # Update stratify column for the second split
        if stratify_col is not None:
            stratify_col = train_val_df[stratify]
        
        # Second split: separate validation set from training set
        # Adjust validation_size to be relative to the train_val_df size
        adjusted_val_size = validation_size / (1 - test_size)
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=adjusted_val_size, 
            random_state=random_state, stratify=stratify_col
        )
        
        logger.debug(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def load_target_datasets(self, base_dir: str, targets: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load datasets for multiple target molecules.
        
        Parameters
        ----------
        base_dir : str
            Base directory containing target-specific datasets
        targets : List[str]
            List of target names
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping target names to their respective DataFrames
        """
        logger.info(f"Loading datasets for targets: {targets}")
        target_dfs = {}
        
        for target in targets:
            target_path = os.path.join(base_dir, f"{target.lower()}.csv")
            
            try:
                if os.path.exists(target_path):
                    target_dfs[target] = self.load_from_csv(target_path)
                    logger.debug(f"Loaded data for target: {target} ({len(target_dfs[target])} rows)")
                else:
                    logger.warning(f"Dataset for target '{target}' not found at {target_path}")
            except Exception as e:
                logger.error(f"Error loading dataset for target '{target}': {str(e)}")
        
        return target_dfs
    
    def combine_datasets(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple target datasets into a single DataFrame.
        
        Parameters
        ----------
        dfs : Dict[str, pd.DataFrame]
            Dictionary of target-specific DataFrames
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame
        """
        if not dfs:
            logger.warning("No datasets to combine")
            return pd.DataFrame()
        
        combined = []
        
        for target, df in dfs.items():
            # Add target column if not already present
            if 'Target_Name' not in df.columns:
                df = df.copy()
                df['Target_Name'] = target
            
            combined.append(df)
        
        result = pd.concat(combined, ignore_index=True)
        logger.info(f"Combined {len(dfs)} datasets with total {len(result)} rows")
        
        return result
