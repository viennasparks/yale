import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional

class AptamerPreprocessor:
    """
    Class for preprocessing and cleaning aptamer data.
    """
    
    def __init__(self):
        """
        Initialize the AptamerPreprocessor.
        """
        pass
    
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
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Remove duplicate sequences
        cleaned_df.drop_duplicates(subset=["Sequence"], inplace=True)
        
        # Standardize sequences to uppercase
        cleaned_df["Sequence"] = cleaned_df["Sequence"].str.upper()
        
        # Handle missing values
        if "GC_Content" in cleaned_df.columns:
            cleaned_df["GC_Content"].fillna(
                cleaned_df["Sequence"].apply(self._calculate_gc_content), 
                inplace=True
            )
        
        # Remove sequences with non-standard nucleotides or invalid characters
        valid_nucleotides = set("ATGCU")
        cleaned_df = cleaned_df[
            cleaned_df["Sequence"].apply(
                lambda seq: all(nucleotide in valid_nucleotides for nucleotide in seq)
            )
        ]
        
        # Reset index after dropping rows
        cleaned_df.reset_index(drop=True, inplace=True)
        
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
        sequence = sequence.upper()
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
        normalized_df = df.copy()
        
        # Define mappings for target name normalization
        target_mappings = {
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
        
        # Apply the mapping to normalize target names
        if "Target_Name" in normalized_df.columns:
            normalized_df["Target_Name"] = normalized_df["Target_Name"].apply(
                lambda x: target_mappings.get(x, x)
            )
        
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
            raise ValueError("DataFrame does not contain a 'Target_Name' column")
        
        # Normalize the target name for consistency
        normalized_target = target.upper().replace(" ", "_").replace("-", "_")
        
        # Apply the filter
        filtered_df = df[df["Target_Name"].str.upper().str.replace(" ", "_").str.replace("-", "_") == normalized_target]
        
        if filtered_df.empty:
            print(f"Warning: No data found for target '{target}'")
        
        return filtered_df
