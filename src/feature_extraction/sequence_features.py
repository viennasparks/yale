import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import re
from collections import Counter
from loguru import logger
from itertools import product
from sklearn.preprocessing import StandardScaler

class SequenceFeatureExtractor:
    """
    Extract features from aptamer sequences for machine learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SequenceFeatureExtractor.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        self.k_mer_sizes = self.config.get('k_mer_sizes', [1, 2, 3])
        self.include_composition = self.config.get('include_composition_features', True)
        self.include_position_features = self.config.get('include_position_features', True)
        self.include_physicochemical = self.config.get('include_physicochemical_features', True)
        
        self.nucleotide_mapping = {
            'A': [1, 0, 0, 0, 0],  # Adenine
            'T': [0, 1, 0, 0, 0],  # Thymine
            'G': [0, 0, 1, 0, 0],  # Guanine
            'C': [0, 0, 0, 1, 0],  # Cytosine
            'U': [0, 0, 0, 0, 1],  # Uracil (for RNA)
            'N': [0.25, 0.25, 0.25, 0.25, 0]  # Unknown (average of ATGC)
        }
        
        # Physicochemical properties based on literature
        self.physicochemical_properties = {
            'A': {'hydrophobicity': -0.5, 'stacking_energy': -1.0, 'hydrogen_bond_donor': 1, 'hydrogen_bond_acceptor': 1},
            'T': {'hydrophobicity': 0.6, 'stacking_energy': -0.9, 'hydrogen_bond_donor': 2, 'hydrogen_bond_acceptor': 2},
            'G': {'hydrophobicity': 0.1, 'stacking_energy': -1.3, 'hydrogen_bond_donor': 2, 'hydrogen_bond_acceptor': 1},
            'C': {'hydrophobicity': 0.1, 'stacking_energy': -0.8, 'hydrogen_bond_donor': 1, 'hydrogen_bond_acceptor': 3},
            'U': {'hydrophobicity': 0.6, 'stacking_energy': -0.9, 'hydrogen_bond_donor': 2, 'hydrogen_bond_acceptor': 2},
        }
        
        logger.info("Initialized SequenceFeatureExtractor")
    
    def extract_all_features(self, sequences: List[str]) -> pd.DataFrame:
        """
        Extract all sequence features for a list of aptamer sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all extracted features
        """
        if not sequences:
            logger.warning("No sequences provided for feature extraction")
            return pd.DataFrame()
        
        logger.info(f"Extracting sequence features for {len(sequences)} sequences")
        
        # Start with basic features
        feature_dict = {
            'sequence': sequences,
            'length': [len(seq) for seq in sequences],
            'gc_content': [self.calculate_gc_content(seq) for seq in sequences]
        }
        
        # Extract nucleotide composition features
        if self.include_composition:
            logger.debug("Extracting nucleotide composition features")
            composition_features = self.extract_nucleotide_composition(sequences)
            feature_dict.update(composition_features)
        
        # Extract k-mer features
        logger.debug(f"Extracting k-mer features for k={self.k_mer_sizes}")
        kmer_features = self.extract_kmer_features(sequences, k_values=self.k_mer_sizes)
        
        for k, k_features in kmer_features.items():
            logger.debug(f"Adding {k_features.shape[1]} features for {k}-mers")
            for col_idx in range(k_features.shape[1]):
                feature_dict[f'{k}mer_{col_idx}'] = k_features[:, col_idx]
        
        # Extract physicochemical properties
        if self.include_physicochemical:
            logger.debug("Extracting physicochemical features")
            pc_features = self.extract_physicochemical_features(sequences)
            feature_dict.update(pc_features)
        
        # Extract positional features if configured
        if self.include_position_features:
            logger.debug("Extracting position-specific features")
            pos_features = self.extract_position_features(sequences)
            feature_dict.update(pos_features)
        
        result_df = pd.DataFrame(feature_dict)
        logger.info(f"Extracted {len(result_df.columns) - 1} features (excluding sequence column)")
        
        return result_df
    
    def calculate_gc_content(self, sequence: str) -> float:
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
    
    def extract_nucleotide_composition(self, sequences: List[str]) -> Dict[str, List[float]]:
        """
        Extract nucleotide composition features.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary of nucleotide composition features
        """
        features = {}
        nucleotides = ['A', 'T', 'G', 'C', 'U']
        
        # Initialize feature lists
        for nt in nucleotides:
            features[f'{nt}_count'] = []
            features[f'{nt}_percentage'] = []
        
        # Calculate counts and percentages
        for seq in sequences:
            seq = seq.upper()
            length = len(seq) if len(seq) > 0 else 1  # Avoid division by zero
            
            for nt in nucleotides:
                count = seq.count(nt)
                features[f'{nt}_count'].append(count)
                features[f'{nt}_percentage'].append((count / length) * 100)
        
        # Calculate dinucleotide frequencies
        dinucleotides = [''.join(nt) for nt in product(['A', 'T', 'G', 'C'], repeat=2)]
        
        for di in dinucleotides:
            features[f'di_{di}'] = []
            
        for seq in sequences:
            seq = seq.upper()
            total_dinucleotides = max(len(seq) - 1, 1)  # Avoid division by zero
            
            for di in dinucleotides:
                # Count overlapping instances
                count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == di)
                features[f'di_{di}'].append(count / total_dinucleotides)
        
        return features
    
    def extract_kmer_features(self, sequences: List[str], 
                             k_values: List[int] = [1, 2, 3]) -> Dict[int, np.ndarray]:
        """
        Extract k-mer frequency features from sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
        k_values : List[int], optional
            List of k values to use for k-mer extraction, by default [1, 2, 3]
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping k values to arrays of k-mer frequencies
        """
        result = {}
        
        for k in k_values:
            # Generate all possible k-mers for DNA
            possible_kmers = [''.join(p) for p in product('ATGC', repeat=k)]
            
            # Initialize feature matrix
            feature_matrix = np.zeros((len(sequences), len(possible_kmers)))
            kmer_to_index = {kmer: i for i, kmer in enumerate(possible_kmers)}
            
            # Calculate k-mer frequencies for each sequence
            for seq_idx, seq in enumerate(sequences):
                seq = seq.upper()
                total_kmers = max(len(seq) - k + 1, 1)  # Avoid division by zero
                
                kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
                kmer_counts = Counter(kmers)
                
                # Fill the feature matrix
                for kmer, count in kmer_counts.items():
                    if kmer in kmer_to_index:  # Ignore kmers with non-standard nucleotides
                        feature_matrix[seq_idx, kmer_to_index[kmer]] = count / total_kmers
            
            result[k] = feature_matrix
        
        return result
    
    def extract_physicochemical_features(self, sequences: List[str]) -> Dict[str, List[float]]:
        """
        Extract physicochemical properties from sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary of physicochemical features
        """
        features = {}
        properties = ['hydrophobicity', 'stacking_energy', 'hydrogen_bond_donor', 'hydrogen_bond_acceptor']
        
        # Initialize features
        for prop in properties:
            features[f'avg_{prop}'] = []
            features[f'max_{prop}'] = []
            features[f'min_{prop}'] = []
            features[f'range_{prop}'] = []
        
        for seq in sequences:
            seq = seq.upper()
            
            for prop in properties:
                # Extract property values for each nucleotide in the sequence
                prop_values = [self.physicochemical_properties.get(nt, {}).get(prop, 0) for nt in seq if nt in 'ATGCU']
                
                if prop_values:
                    features[f'avg_{prop}'].append(np.mean(prop_values))
                    features[f'max_{prop}'].append(np.max(prop_values))
                    features[f'min_{prop}'].append(np.min(prop_values))
                    features[f'range_{prop}'].append(np.max(prop_values) - np.min(prop_values))
                else:
                    # Default values if no valid nucleotides
                    features[f'avg_{prop}'].append(0)
                    features[f'max_{prop}'].append(0)
                    features[f'min_{prop}'].append(0)
                    features[f'range_{prop}'].append(0)
        
        # Add purine/pyrimidine ratio
        features['purine_pyrimidine_ratio'] = []
        
        for seq in sequences:
            seq = seq.upper()
            purines = seq.count('A') + seq.count('G')
            pyrimidines = seq.count('T') + seq.count('C') + seq.count('U')
            
            ratio = purines / pyrimidines if pyrimidines > 0 else 0
            features['purine_pyrimidine_ratio'].append(ratio)
        
        return features
    
    def extract_position_features(self, sequences: List[str], normalize: bool = True) -> Dict[str, List[float]]:
        """
        Extract position-specific features from sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
        normalize : bool, optional
            Whether to normalize the features, by default True
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary of position-specific features
        """
        # Find the maximum sequence length, but cap it to reduce dimensionality
        max_length = min(max(len(seq) for seq in sequences), 100)
        logger.debug(f"Using max_length={max_length} for position features")
        
        features = {}
        
        # Add 5' and 3' end features (first and last 5 nucleotides)
        end_length = 5
        
        # 5' end features
        for pos in range(end_length):
            for nt in 'ATGC':
                features[f'5p_{pos+1}_{nt}'] = []
            
            for seq in sequences:
                seq = seq.upper()
                for nt in 'ATGC':
                    if pos < len(seq):
                        features[f'5p_{pos+1}_{nt}'].append(1 if seq[pos] == nt else 0)
                    else:
                        features[f'5p_{pos+1}_{nt}'].append(0)
        
        # 3' end features
        for pos in range(end_length):
            for nt in 'ATGC':
                features[f'3p_{pos+1}_{nt}'] = []
            
            for seq in sequences:
                seq = seq.upper()
                for nt in 'ATGC':
                    if len(seq) > pos:
                        features[f'3p_{pos+1}_{nt}'].append(1 if seq[-(pos+1)] == nt else 0)
                    else:
                        features[f'3p_{pos+1}_{nt}'].append(0)
        
        # Add nucleotide run features (e.g., poly-A, poly-G)
        for nt in 'ATGC':
            features[f'max_run_{nt}'] = []
            
            for seq in sequences:
                seq = seq.upper()
                # Find the longest run of this nucleotide
                max_run = 0
                current_run = 0
                
                for char in seq:
                    if char == nt:
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 0
                
                features[f'max_run_{nt}'].append(max_run)
        
        # Normalize features if requested
        if normalize:
            scaler = StandardScaler()
            for feature_name, values in features.items():
                if len(values) > 0:
                    features[feature_name] = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten().tolist()
        
        return features
    
    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """
        One-hot encode a nucleotide sequence.
        
        Parameters
        ----------
        sequence : str
            Nucleotide sequence
            
        Returns
        -------
        np.ndarray
            One-hot encoded sequence
        """
        sequence = sequence.upper()
        encoded = []
        
        for nucleotide in sequence:
            if nucleotide in self.nucleotide_mapping:
                encoded.append(self.nucleotide_mapping[nucleotide])
            else:
                # For non-standard nucleotides, use the encoding for 'N'
                encoded.append(self.nucleotide_mapping['N'])
        
        return np.array(encoded)
    
    def extract_position_specific_matrix(self, sequences: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """
        Extract position-specific scoring matrix (PSSM) from sequences.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
        max_length : Optional[int], optional
            Maximum length to consider, by default None (uses the longest sequence)
            
        Returns
        -------
        np.ndarray
            Position-specific scoring matrix with shape (max_length, 5)
            representing probabilities of each nucleotide at each position
        """
        # Determine maximum sequence length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Initialize PSSM with pseudocounts (adding 0.01 to avoid log(0))
        pssm = np.ones((max_length, 5)) * 0.01
        
        # Count nucleotide occurrences at each position
        for seq in sequences:
            seq = seq.upper()
            
            for pos, nt in enumerate(seq):
                if pos >= max_length:
                    break
                
                if nt == 'A':
                    pssm[pos, 0] += 1
                elif nt == 'T':
                    pssm[pos, 1] += 1
                elif nt == 'G':
                    pssm[pos, 2] += 1
                elif nt == 'C':
                    pssm[pos, 3] += 1
                elif nt == 'U':
                    pssm[pos, 4] += 1
        
        # Normalize to get probabilities
        row_sums = pssm.sum(axis=1, keepdims=True)
        pssm = pssm / row_sums
        
        return pssm
    
    def calculate_structural_tendency_scores(self, sequences: List[str]) -> pd.DataFrame:
        """
        Calculate structural tendency scores for each position in the sequences.
        This is a simple proxy for structural propensity without full structure prediction.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        pd.DataFrame
            DataFrame with structural tendency scores
        """
        # Dictionary of base-pairing scores (higher means more likely to form pairs)
        pairing_scores = {
            'A': 0.2,  # Weak pairing (only pairs with U in RNA)
            'T': 0.3,  # Moderate pairing (pairs with A)
            'U': 0.3,  # Moderate pairing (pairs with A)
            'G': 0.5,  # Strong pairing (pairs with C, and can form G-G pairs)
            'C': 0.5   # Strong pairing (pairs with G)
        }
        
        # Calculate scores for each sequence
        scores = []
        for seq in sequences:
            seq = seq.upper()
            seq_scores = [pairing_scores.get(nt, 0) for nt in seq]
            
            # Calculate running average to account for neighboring effects
            window_size = 3
            smoothed_scores = []
            for i in range(len(seq_scores)):
                start = max(0, i - window_size // 2)
                end = min(len(seq_scores), i + window_size // 2 + 1)
                smoothed_scores.append(np.mean(seq_scores[start:end]))
            
            scores.append(smoothed_scores)
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'sequence': sequences,
            'structural_scores': scores
        })
        
        return result_df
    
    def g_quadruplex_potential(self, sequences: List[str]) -> List[float]:
        """
        Calculate G-quadruplex forming potential for sequences.
        G-quadruplexes are stable structures formed in G-rich regions and can affect aptamer binding.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        List[float]
            G-quadruplex potential scores
        """
        # G-quadruplex pattern: G-rich pattern with 2+ consecutive Gs
        # More sophisticated G4 predictors look for G3+N1-7G3+N1-7G3+N1-7G3+
        g4_pattern = r'(G{2,})(.{1,7})(G{2,})(.{1,7})(G{2,})(.{1,7})(G{2,})'
        strict_pattern = r'(G{3,})(.{1,7})(G{3,})(.{1,7})(G{3,})(.{1,7})(G{3,})'
        
        scores = []
        
        for seq in sequences:
            seq = seq.upper()
            
            # Check for G4 patterns
            g4_matches = re.findall(g4_pattern, seq)
            strict_matches = re.findall(strict_pattern, seq)
            
            # Calculate G-richness
            g_count = seq.count('G')
            g_percentage = (g_count / len(seq)) * 100 if len(seq) > 0 else 0
            
            # Calculate G4 potential score (combining pattern matches and G-richness)
            score = len(g4_matches) * 0.5 + len(strict_matches) * 1.0 + g_percentage * 0.02
            
            scores.append(score)
        
        return scores
