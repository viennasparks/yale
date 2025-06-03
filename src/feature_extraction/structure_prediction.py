import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import tempfile
import subprocess
from io import StringIO
from loguru import logger
import RNA  # ViennaRNA package
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class StructurePredictor:
    """
    Predict and analyze aptamer secondary structures using thermodynamic models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StructurePredictor.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        
        # Configuration parameters with defaults
        self.algorithm = self.config.get('algorithm', 'mfe')  # mfe, centroid, suboptimal
        self.energy_model = self.config.get('energy_model', 'turner2004')
        self.temperature = float(self.config.get('temperature', 37.0))
        self.max_suboptimal = int(self.config.get('max_suboptimal_structures', 10))
        self.energy_range = float(self.config.get('energy_range', 5.0))
        self.dangling_ends = int(self.config.get('dangling_ends', 2))
        
        # Initialize ViennaRNA model details
        self.model_details = RNA.md()
        self.model_details.temperature = self.temperature
        self.model_details.dangles = self.dangling_ends
        self.model_details.energy_set = 0  # Default (Turner 2004)
        
        logger.info(f"Initialized StructurePredictor with algorithm={self.algorithm}, "
                   f"temperature={self.temperature}°C, energy_model={self.energy_model}")
    
    def predict_mfe_structure(self, sequence: str) -> Tuple[str, float]:
        """
        Predict the minimum free energy structure for an aptamer sequence.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        Tuple[str, float]
            (Structure in dot-bracket notation, MFE value)
        """
        # Convert DNA to RNA for ViennaRNA
        sequence = sequence.upper().replace('T', 'U')
        
        try:
            # Create a fold compound with the specified model details
            fc = RNA.fold_compound(sequence, self.model_details)
            
            # Calculate MFE structure
            structure, mfe = fc.mfe()
            
            logger.debug(f"Predicted MFE structure for seq length {len(sequence)}, energy: {mfe:.2f} kcal/mol")
            return structure, mfe
        except Exception as e:
            logger.error(f"Error predicting MFE structure: {str(e)}")
            # Return a simple unpaired structure as fallback
            return '.' * len(sequence), 0.0
    
    def predict_centroid_structure(self, sequence: str) -> Tuple[str, float]:
        """
        Predict the centroid structure based on the ensemble of possible structures.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        Tuple[str, float]
            (Centroid structure in dot-bracket notation, ensemble free energy)
        """
        # Convert DNA to RNA for ViennaRNA
        sequence = sequence.upper().replace('T', 'U')
        
        try:
            # Create a fold compound with the specified model details
            fc = RNA.fold_compound(sequence, self.model_details)
            
            # Calculate partition function
            _, ensemble_energy = fc.pf()
            
            # Calculate centroid structure
            centroid_structure, distance = fc.centroid()
            
            logger.debug(f"Predicted centroid structure, ensemble energy: {ensemble_energy:.2f} kcal/mol")
            return centroid_structure, ensemble_energy
        except Exception as e:
            logger.error(f"Error predicting centroid structure: {str(e)}")
            # Return a simple unpaired structure as fallback
            return '.' * len(sequence), 0.0
    
    def predict_suboptimal_structures(self, sequence: str) -> List[Tuple[str, float]]:
        """
        Predict suboptimal structures within a specified energy range.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (structure in dot-bracket notation, energy) tuples
        """
        # Convert DNA to RNA for ViennaRNA
        sequence = sequence.upper().replace('T', 'U')
        
        try:
            # Create a fold compound with the specified model details
            fc = RNA.fold_compound(sequence, self.model_details)
            
            # Get suboptimal structures
            subopt_structures = fc.subopt(self.energy_range)
            
            # Return only the specified maximum number of structures
            limited_structures = [(s.structure, s.energy) for s in subopt_structures[:self.max_suboptimal]]
            
            logger.debug(f"Predicted {len(limited_structures)} suboptimal structures within "
                        f"{self.energy_range} kcal/mol of MFE")
            
            return limited_structures
        except Exception as e:
            logger.error(f"Error predicting suboptimal structures: {str(e)}")
            # Return a simple unpaired structure as fallback
            return [('.' * len(sequence), 0.0)]
    
    def predict_ensemble_properties(self, sequence: str) -> Dict[str, Any]:
        """
        Predict properties of the Boltzmann ensemble of structures.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of ensemble properties
        """
        # Convert DNA to RNA for ViennaRNA
        sequence = sequence.upper().replace('T', 'U')
        
        try:
            # Create a fold compound with the specified model details
            fc = RNA.fold_compound(sequence, self.model_details)
            
            # Calculate partition function
            _, ensemble_free_energy = fc.pf()
            
            # Calculate ensemble diversity (mean base pair distance in ensemble)
            ensemble_diversity = fc.mean_bp_distance()
            
            # Calculate centroid structure
            centroid_structure, distance = fc.centroid()
            
            # Calculate positional entropy
            positional_entropy = fc.positional_entropy()
            
            return {
                "ensemble_free_energy": ensemble_free_energy,
                "ensemble_diversity": ensemble_diversity,
                "centroid_structure": centroid_structure,
                "mean_positional_entropy": np.mean(positional_entropy) if len(positional_entropy) > 0 else 0,
                "max_positional_entropy": np.max(positional_entropy) if len(positional_entropy) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating ensemble properties: {str(e)}")
            # Return default values as fallback
            return {
                "ensemble_free_energy": 0.0,
                "ensemble_diversity": 0.0,
                "centroid_structure": '.' * len(sequence),
                "mean_positional_entropy": 0.0,
                "max_positional_entropy": 0.0
            }
    
    def calculate_structural_features(self, structure: str) -> Dict[str, float]:
        """
        Calculate features from a secondary structure in dot-bracket notation.
        
        Parameters
        ----------
        structure : str
            Structure in dot-bracket notation
            
        Returns
        -------
        Dict[str, float]
            Dictionary of structural features
        """
        # Count structural elements
        unpaired_count = structure.count('.')
        paired_count = structure.count('(')  # or count of ')' which should be equal
        
        # Calculate percentages
        total_length = len(structure)
        unpaired_percentage = (unpaired_count / total_length) * 100 if total_length > 0 else 0
        paired_percentage = (paired_count * 2 / total_length) * 100 if total_length > 0 else 0
        
        # Identify stems, loops, and bulges
        stems = []
        hairpin_loops = []
        internal_loops = []
        bulges = []
        
        # Parse structure to identify elements
        stack = []
        stem_start_positions = []
        
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
                stem_start_positions.append(i)
            elif char == ')':
                if stack:
                    start_pos = stack.pop()
                    
                    # Check if this is a hairpin loop
                    if not any(c == '(' for c in structure[start_pos+1:i]):
                        # Found a hairpin loop
                        stem_length = 1  # Count this base pair
                        loop_length = i - start_pos - 1
                        
                        # Look for more base pairs in this stem
                        j = 1
                        while (start_pos - j >= 0 and i + j < len(structure) and 
                               structure[start_pos - j] == '(' and structure[i + j] == ')'):
                            stem_length += 1
                            j += 1
                        
                        stems.append(stem_length)
                        hairpin_loops.append(loop_length)
        
        # Find bulges and internal loops
        # This is a simplified approach; real RNA structure parsers use more complex algorithms
        i = 0
        while i < len(structure):
            if structure[i] == '(':
                # Found start of a stem
                j = i + 1
                bulge_size = 0
                
                while j < len(structure) and structure[j] == '.':
                    bulge_size += 1
                    j += 1
                
                if bulge_size > 0 and j < len(structure) and structure[j] == '(':
                    # Found a bulge
                    bulges.append(bulge_size)
            
            i += 1
        
        # Calculate average lengths
        avg_stem_length = np.mean(stems) if stems else 0
        avg_hairpin_loop_length = np.mean(hairpin_loops) if hairpin_loops else 0
        avg_bulge_length = np.mean(bulges) if bulges else 0
        
        return {
            "unpaired_percentage": unpaired_percentage,
            "paired_percentage": paired_percentage,
            "stem_count": len(stems),
            "hairpin_loop_count": len(hairpin_loops),
            "bulge_count": len(bulges),
            "avg_stem_length": avg_stem_length,
            "avg_hairpin_loop_length": avg_hairpin_loop_length, 
            "avg_bulge_length": avg_bulge_length,
            "max_stem_length": max(stems) if stems else 0,
            "structure_complexity_score": len(stems) * 2 + len(hairpin_loops) + len(bulges) * 0.5
        }
    
    def predict_and_analyze_structures(self, sequences: List[str], parallel: bool = True) -> pd.DataFrame:
        """
        Predict structures for a list of sequences and extract structural features.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
        parallel : bool, optional
            Whether to predict structures in parallel, by default True
            
        Returns
        -------
        pd.DataFrame
            DataFrame with structural predictions and features
        """
        results = []
        
        if parallel and len(sequences) > 1:
            logger.info(f"Predicting structures for {len(sequences)} sequences in parallel")
            with ThreadPoolExecutor() as executor:
                future_to_sequence = {
                    executor.submit(self._process_single_sequence, seq): seq for seq in sequences
                }
                
                for future in as_completed(future_to_sequence):
                    results.append(future.result())
        else:
            logger.info(f"Predicting structures for {len(sequences)} sequences sequentially")
            for seq in sequences:
                results.append(self._process_single_sequence(seq))
        
        return pd.DataFrame(results)
    
    def _process_single_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Process a single sequence to predict structures and extract features.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with structural predictions and features
        """
        # Choose prediction algorithm based on configuration
        if self.algorithm == 'centroid':
            structure, energy = self.predict_centroid_structure(sequence)
        elif self.algorithm == 'suboptimal':
            structures = self.predict_suboptimal_structures(sequence)
            # Use the first (MFE) structure
            structure, energy = structures[0] if structures else ('.' * len(sequence), 0.0)
        else:  # default to MFE
            structure, energy = self.predict_mfe_structure(sequence)
        
        # Calculate ensemble properties
        ensemble_properties = self.predict_ensemble_properties(sequence)
        
        # Calculate structural features
        structural_features = self.calculate_structural_features(structure)
        
        return {
            "sequence": sequence,
            "predicted_structure": structure,
            "energy": energy,
            **ensemble_properties,
            **structural_features
        }
    
    def get_g_quadruplex_propensity(self, sequences: List[str]) -> List[float]:
        """
        Calculate G-quadruplex forming propensity for each sequence.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        List[float]
            G-quadruplex propensity scores
        """
        # Regular expression pattern for G-quadruplex motifs
        # (G{3,}).{1,7}(G{3,}).{1,7}(G{3,}).{1,7}(G{3,})
        g4_pattern = r'(G{3,}).{1,7}(G{3,}).{1,7}(G{3,}).{1,7}(G{3,})'
        
        scores = []
        
        for seq in sequences:
            seq = seq.upper()
            # Count G-quadruplex motifs
            g4_motifs = len(re.findall(g4_pattern, seq))
            
            # Calculate G content
            g_content = seq.count('G') / len(seq) if len(seq) > 0 else 0
            
            # Calculate a score based on motif presence and G content
            g4_score = g4_motifs * 2.0 + g_content * 0.5
            
            scores.append(g4_score)
        
        return scores
    
    def calculate_thermodynamic_stability(self, sequences: List[str]) -> List[Dict[str, float]]:
        """
        Calculate thermodynamic stability metrics for each sequence.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
            
        Returns
        -------
        List[Dict[str, float]]
            List of dictionaries containing thermodynamic metrics
        """
        results = []
        
        for seq in sequences:
            try:
                # Convert to RNA
                seq_rna = seq.upper().replace('T', 'U')
                
                # Create fold compound with model details
                fc = RNA.fold_compound(seq_rna, self.model_details)
                
                # Calculate MFE
                structure, mfe = fc.mfe()
                
                # Calculate partition function for ensemble energy
                _, efe = fc.pf()
                
                # Calculate ensemble diversity
                ed = fc.mean_bp_distance()
                
                # Calculate melting temperature (approximation)
                # This is a simplified approach; real Tm calculation would use specific salt conditions
                gc_content = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
                approx_tm = 64.9 + 41.0 * (gc_content - 0.34)
                
                # Adjusted Tm based on MFE
                adjusted_tm = approx_tm - (abs(mfe) * 0.3)
                
                results.append({
                    "mfe": mfe,
                    "ensemble_free_energy": efe,
                    "ensemble_diversity": ed,
                    "approximated_tm": adjusted_tm, 
                    "stability_score": abs(mfe) - (ed * 0.1)  # Higher is more stable
                })
            except Exception as e:
                logger.error(f"Error calculating thermodynamic stability: {str(e)}")
                results.append({
                    "mfe": 0.0,
                    "ensemble_free_energy": 0.0,
                    "ensemble_diversity": 0.0,
                    "approximated_tm": 0.0,
                    "stability_score": 0.0
                })
        
        return results
