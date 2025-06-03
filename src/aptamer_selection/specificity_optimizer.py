import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import random
from loguru import logger
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from src.feature_extraction.structure_prediction import StructurePredictor
from src.models.binding_affinity import BindingAffinityPredictor
from src.models.cross_reactivity import CrossReactivityAnalyzer

class AptamerOptimizer:
    """
    Optimize aptamer sequences to improve specificity and reduce cross-reactivity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AptamerOptimizer.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        
        # Optimization parameters
        self.optimization_iterations = int(self.config.get('optimization_iterations', 1000))
        self.population_size = int(self.config.get('population_size', 200))
        self.mutation_rate = float(self.config.get('mutation_rate', 0.05))
        self.crossover_rate = float(self.config.get('crossover_rate', 0.8))
        
        # Model instances
        self.structure_predictor = StructurePredictor(self.config.get('structure_prediction', {}))
        self.binding_model = None
        self.cross_reactivity_model = None
        
        # Weights for fitness calculation
        self.specificity_weight = float(self.config.get('specificity_weight', 0.6))
        self.binding_affinity_weight = float(self.config.get('binding_affinity_weight', 0.4))
        self.structural_stability_weight = float(self.config.get('structural_stability_weight', 0.3))
        
        logger.info(f"Initialized AptamerOptimizer with iterations={self.optimization_iterations}, "
                   f"population_size={self.population_size}, mutation_rate={self.mutation_rate}")
    
    def optimize_aptamers(self, initial_aptamers: pd.DataFrame, 
                         target: str,
                         non_targets: List[str],
                         num_optimized: int = 10) -> pd.DataFrame:
        """
        Optimize aptamer sequences to maximize specificity for a target and minimize cross-reactivity.
        
        Parameters
        ----------
        initial_aptamers : pd.DataFrame
            DataFrame with initial aptamer sequences
        target : str
            Target molecule name
        non_targets : List[str]
            List of non-target molecules to avoid cross-reactivity with
        num_optimized : int, optional
            Number of optimized aptamers to return, by default 10
            
        Returns
        -------
        pd.DataFrame
            DataFrame with optimized aptamer sequences
        """
        logger.info(f"Starting optimization for target '{target}' vs {len(non_targets)} non-targets")
        
        # Initialize models if not already initialized
        if self.binding_model is None:
            self.binding_model = BindingAffinityPredictor(self.config.get('binding_affinity', {}))
        
        if self.cross_reactivity_model is None:
            self.cross_reactivity_model = CrossReactivityAnalyzer(self.config.get('cross_reactivity', {}))
        
        # Extract sequences and initialize population
        if 'Sequence' in initial_aptamers.columns:
            sequences = initial_aptamers['Sequence'].tolist()
        else:
            sequences = initial_aptamers['sequence'].tolist()
        
        # Start with a population from the initial aptamers
        population = sequences.copy()
        
        # If we need more for the initial population, generate variations
        while len(population) < self.population_size:
            # Add mutations of existing sequences
            idx = random.randint(0, len(sequences) - 1)
            mutated = self._mutate_sequence(sequences[idx])
            population.append(mutated)
        
        # Truncate if we have too many
        population = population[:self.population_size]
        
        # Calculate initial fitness
        fitness_scores = self._evaluate_population(population, target, non_targets)
        
        # Main optimization loop
        best_fitness = max(fitness_scores)
        best_sequence = population[fitness_scores.index(best_fitness)]
        
        logger.info(f"Initial population best fitness: {best_fitness:.4f}")
        
        for iteration in tqdm(range(self.optimization_iterations), desc="Optimizing aptamers"):
            # Select parents based on fitness
            parents = self._select_parents(population, fitness_scores)
            
            # Create next generation
            next_generation = self._create_next_generation(parents)
            
            # Evaluate new population
            new_fitness_scores = self._evaluate_population(next_generation, target, non_targets)
            
            # Update best solution if improved
            max_new_fitness = max(new_fitness_scores)
            if max_new_fitness > best_fitness:
                best_fitness = max_new_fitness
                best_idx = new_fitness_scores.index(max_new_fitness)
                best_sequence = next_generation[best_idx]
                
                logger.debug(f"Iteration {iteration+1}: New best fitness = {best_fitness:.4f}")
            
            # Update population for next iteration
            population = next_generation
            fitness_scores = new_fitness_scores
            
            # Early stopping if we reach very high fitness
            if best_fitness > 0.98:
                logger.info(f"Early stopping at iteration {iteration+1} with fitness = {best_fitness:.4f}")
                break
        
        # Select best aptamers from the final population
        combined = list(zip(population, fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        best_aptamers = combined[:num_optimized]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'Sequence': [apt[0] for apt in best_aptamers],
            'Target_Name': target,
            'optimization_fitness': [apt[1] for apt in best_aptamers],
            'length': [len(apt[0]) for apt in best_aptamers]
        })
        
        # Add additional predictions
        self._add_predictions(result, target, non_targets)
        
        logger.info(f"Optimization complete. Best fitness achieved: {best_fitness:.4f}")
        logger.info(f"Returning {len(result)} optimized aptamers")
        
        return result
    
    def _evaluate_population(self, population: List[str], 
                           target: str, 
                           non_targets: List[str]) -> List[float]:
        """
        Evaluate the fitness of a population of aptamer sequences.
        
        Parameters
        ----------
        population : List[str]
            List of aptamer sequences
        target : str
            Target molecule name
        non_targets : List[str]
            List of non-target molecules to avoid cross-reactivity with
            
        Returns
        -------
        List[float]
            List of fitness scores
        """
        # Create a temporary DataFrame for evaluation
        eval_df = pd.DataFrame({
            'Sequence': population,
            'sequence': population,  # Include both column names for compatibility
            'Target_Name': target
        })
        
        # Evaluate binding affinity (higher is better)
        try:
            affinity_scores = self.binding_model.predict(eval_df)
            # Normalize to [0, 1]
            min_aff = min(affinity_scores)
            max_aff = max(affinity_scores)
            if max_aff > min_aff:
                normalized_affinity = [(a - min_aff) / (max_aff - min_aff) for a in affinity_scores]
            else:
                normalized_affinity = [0.5] * len(affinity_scores)
        except Exception as e:
            logger.warning(f"Error evaluating binding affinity: {str(e)}")
            normalized_affinity = [0.5] * len(population)  # Default score
        
        # Evaluate structure stability
        stability_scores = []
        for seq in population:
            stability_scores.append(self._evaluate_structure_stability(seq))
        
        # Evaluate cross-reactivity (lower is better)
        try:
            # Create artificial dataset for cross-reactivity evaluation
            cross_react_data = []
            
            # Add target sequences
            for seq in population:
                cross_react_data.append({
                    'Sequence': seq,
                    'sequence': seq,
                    'Target_Name': target
                })
            
            # Add sequences with non-targets (simulated)
            for non_target in non_targets:
                for seq in population[:10]:  # Use a subset for efficiency
                    cross_react_data.append({
                        'Sequence': seq,
                        'sequence': seq,
                        'Target_Name': non_target
                    })
            
            cross_df = pd.DataFrame(cross_react_data)
            
            # Train cross-reactivity model on this artificial data
            self.cross_reactivity_model.train_cross_reactivity_model(cross_df)
            
            # Predict cross-reactivity for target sequences
            target_df = pd.DataFrame({
                'Sequence': population,
                'sequence': population,
                'Target_Name': target
            })
            
            # Get specificity scores
            spec_df = self.cross_reactivity_model.calculate_specificity_score(target_df)
            specificity_scores = spec_df['combined_specificity'].values
        except Exception as e:
            logger.warning(f"Error evaluating cross-reactivity: {str(e)}")
            specificity_scores = [0.5] * len(population)  # Default score
        
        # Calculate combined fitness
        fitness_scores = []
        for i in range(len(population)):
            fitness = (
                self.binding_affinity_weight * normalized_affinity[i] +
                self.specificity_weight * specificity_scores[i] +
                self.structural_stability_weight * stability_scores[i]
            ) / (self.binding_affinity_weight + self.specificity_weight + self.structural_stability_weight)
            
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _evaluate_structure_stability(self, sequence: str) -> float:
        """
        Evaluate the structural stability of an aptamer sequence.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
            
        Returns
        -------
        float
            Stability score [0, 1] (higher is more stable)
        """
        try:
            # Predict structure
            structure, mfe = self.structure_predictor.predict_mfe_structure(sequence)
            
            # Calculate stability metrics
            ensemble_props = self.structure_predictor.predict_ensemble_properties(sequence)
            
            # Calculate stability score
            # Lower (more negative) MFE indicates more stable structure
            mfe_score = min(abs(mfe) / 50.0, 1.0)  # Scale and cap at 1.0
            
            # Lower ensemble diversity indicates more stable structure
            diversity = ensemble_props.get('ensemble_diversity', len(sequence) / 2)
            diversity_score = 1.0 - min(diversity / len(sequence), 1.0)
            
            # Combined stability score
            stability = 0.7 * mfe_score + 0.3 * diversity_score
            
            return stability
        except Exception as e:
            logger.warning(f"Error evaluating structure stability: {str(e)}")
            return 0.5  # Default score
    
    def _select_parents(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """
        Select parents for the next generation using tournament selection.
        
        Parameters
        ----------
        population : List[str]
            Current population
        fitness_scores : List[float]
            Fitness scores for the current population
            
        Returns
        -------
        List[str]
            Selected parents
        """
        parents = []
        tournament_size = max(3, int(len(population) * 0.1))
        
        while len(parents) < len(population):
            # Select random candidates for tournament
            candidates = random.sample(range(len(population)), tournament_size)
            
            # Find the best candidate
            best_idx = candidates[0]
            best_fitness = fitness_scores[best_idx]
            
            for idx in candidates[1:]:
                if fitness_scores[idx] > best_fitness:
                    best_idx = idx
                    best_fitness = fitness_scores[idx]
            
            # Add the winner to parents
            parents.append(population[best_idx])
        
        return parents
    
    def _create_next_generation(self, parents: List[str]) -> List[str]:
        """
        Create the next generation through crossover and mutation.
        
        Parameters
        ----------
        parents : List[str]
            Selected parents
            
        Returns
        -------
        List[str]
            Next generation
        """
        next_generation = []
        
        # Always keep the best parent (elitism)
        next_generation.append(parents[0])
        
        while len(next_generation) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Apply crossover with probability crossover_rate
            if random.random() < self.crossover_rate and len(parent1) == len(parent2):
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                child1 = self._mutate_sequence(child1)
            
            if random.random() < self.mutation_rate:
                child2 = self._mutate_sequence(child2)
            
            # Add to next generation
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)
        
        # Ensure we have exactly population_size
        return next_generation[:self.population_size]
    
    def _crossover(self, seq1: str, seq2: str) -> Tuple[str, str]:
        """
        Perform crossover between two aptamer sequences.
        
        Parameters
        ----------
        seq1 : str
            First aptamer sequence
        seq2 : str
            Second aptamer sequence
            
        Returns
        -------
        Tuple[str, str]
            Two child sequences
        """
        if len(seq1) != len(seq2):
            return seq1, seq2  # Can't crossover sequences of different lengths
        
        # Choose random crossover point
        point = random.randint(1, len(seq1) - 1)
        
        # Create children
        child1 = seq1[:point] + seq2[point:]
        child2 = seq2[:point] + seq1[point:]
        
        return child1, child2
    
    def _mutate_sequence(self, sequence: str) -> str:
        """
        Apply random mutations to an aptamer sequence.
        
        Parameters
        ----------
        sequence : str
            Original aptamer sequence
            
        Returns
        -------
        str
            Mutated sequence
        """
        # Convert to list for easier manipulation
        seq_list = list(sequence)
        
        # Number of mutations to apply
        num_mutations = max(1, int(len(sequence) * self.mutation_rate))
        
        # Apply mutations
        for _ in range(num_mutations):
            position = random.randint(0, len(sequence) - 1)
            
            # Current nucleotide
            current = seq_list[position]
            
            # Select a different nucleotide
            nucleotides = [nt for nt in 'ATGC' if nt != current]
            new_nucleotide = random.choice(nucleotides)
            
            # Apply mutation
            seq_list[position] = new_nucleotide
        
        return ''.join(seq_list)
    
    def _add_predictions(self, df: pd.DataFrame, target: str, non_targets: List[str]) -> None:
        """
        Add additional predictions for the optimized aptamers.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with optimized aptamers
        target : str
            Target molecule name
        non_targets : List[str]
            List of non-target molecules
        """
        try:
            # Predict binding affinity
            df['predicted_affinity'] = self.binding_model.predict(df)
            
            # Predict structures
            structures = []
            for seq in df['Sequence'].tolist():
                structure, energy = self.structure_predictor.predict_mfe_structure(seq)
                structures.append(structure)
            
            df['predicted_structure'] = structures
            
            # Predict cross-reactivity
            cr_df = self.cross_reactivity_model.predict_cross_reactivity(df)
            
            # Merge the relevant columns
            cross_cols = [col for col in cr_df.columns if col.endswith('_probability')]
            for col in cross_cols:
                df[col] = cr_df[col]
            
            df['specificity_score'] = cr_df['specificity_score'] if 'specificity_score' in cr_df.columns else 0.5
        except Exception as e:
            logger.warning(f"Error adding predictions to optimized aptamers: {str(e)}")
    
    def run_parallel_optimization(self, initial_aptamers: pd.DataFrame,
                                targets: List[str],
                                num_optimized: int = 5) -> pd.DataFrame:
        """
        Run optimization in parallel for multiple targets.
        
        Parameters
        ----------
        initial_aptamers : pd.DataFrame
            DataFrame with initial aptamer sequences
        targets : List[str]
            List of target molecules
        num_optimized : int, optional
            Number of optimized aptamers per target, by default 5
            
        Returns
        -------
        pd.DataFrame
            DataFrame with optimized aptamers for all targets
        """
        results = []
        
        for target in targets:
            # All other targets are non-targets
            non_targets = [t for t in targets if t != target]
            
            # Filter initial aptamers for this target if possible
            if 'Target_Name' in initial_aptamers.columns:
                target_aptamers = initial_aptamers[initial_aptamers['Target_Name'].str.contains(target, case=False)]
                if len(target_aptamers) == 0:
                    target_aptamers = initial_aptamers  # Fallback to all if none match
            else:
                target_aptamers = initial_aptamers
            
            # Run optimization
            optimized = self.optimize_aptamers(
                target_aptamers,
                target=target,
                non_targets=non_targets,
                num_optimized=num_optimized
            )
            
            results.append(optimized)
        
        # Combine all results
        final_df = pd.concat(results, ignore_index=True)
        return final_df
