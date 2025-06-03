import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional

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
        try:
            df = pd.read_csv(file_path)
            
            if self.validation:
                self._validate_aptamer_data(df)
                
            return df
        except Exception as e:
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
        from Bio import SeqIO
        sequences = []
        
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append({
                    "Sequence_ID": record.id,
                    "Sequence": str(record.seq),
                    "Description": record.description
                })
            
            df = pd.DataFrame(sequences)
            return df
        except Exception as e:
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
        required_columns = ["Sequence", "Target_Name"]
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if sequence column contains valid DNA/RNA sequences
        valid_nucleotides = set("ATGCU")
        invalid_sequences = df["Sequence"].apply(
            lambda seq: not all(nucleotide in valid_nucleotides for nucleotide in seq.upper())
        )
        
        if invalid_sequences.any():
            invalid_indices = df.index[invalid_sequences].tolist()
            raise ValueError(f"Invalid nucleotide sequences detected at indices: {invalid_indices}")
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  random_state: int = 42) -> tuple:
        """
        Split the data into training and testing sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        test_size : float, optional
            Fraction of data to use for testing, by default 0.2
        random_state : int, optional
            Random seed for reproducibility, by default 42
            
        Returns
        -------
        tuple
            (train_df, test_df) - Training and testing DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        return train_df, test_df
