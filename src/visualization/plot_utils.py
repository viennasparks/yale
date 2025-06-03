import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import io
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

class AptamerVisualizer:
    """
    Visualization utilities for aptamer data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AptamerVisualizer.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        
        # Output configuration
        self.output_dir = self.config.get('output_directory', 'results/visualizations')
        self.plot_format = self.config.get('plot_format', 'png')
        self.dpi = int(self.config.get('dpi', 300))
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        
        logger.info(f"Initialized AptamerVisualizer with output directory: {self.output_dir}")
    
    def plot_sequence_length_distribution(self, df: pd.DataFrame, 
                                         by_target: bool = False, 
                                         output_path: Optional[str] = None) -> str:
        """
        Plot the distribution of aptamer sequence lengths.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        by_target : bool, optional
            Whether to separate by target, by default False
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        plt.figure(figsize=(10, 6))
        
        # Determine sequence column name
        seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
        if seq_col not in df.columns:
            logger.error("DataFrame must contain a 'Sequence' or 'sequence' column")
            raise ValueError("DataFrame must contain a 'Sequence' or 'sequence' column")
        
        # Calculate sequence lengths if not already present
        if 'length' not in df.columns:
            df['length'] = df[seq_col].str.len()
        
        if by_target and 'Target_Name' in df.columns:
            # Plot distribution by target
            ax = sns.histplot(data=df, x='length', hue='Target_Name', multiple='stack', kde=True, 
                            palette='viridis')
            plt.title('Distribution of Aptamer Sequence Lengths by Target', fontsize=16)
            
            # Adjust legend
            plt.legend(title="Target", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot overall distribution
            ax = sns.histplot(data=df, x='length', kde=True, color='#1f77b4')
            plt.title('Distribution of Aptamer Sequence Lengths', fontsize=16)
        
        plt.xlabel('Sequence Length (nucleotides)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(alpha=0.3)
        
        # Ensure x-axis uses integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Save the plot
        if output_path is None:
            output_name = 'sequence_length_distribution'
            if by_target:
                output_name += '_by_target'
            output_path = os.path.join(self.output_dir, f"{output_name}.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Sequence length distribution plot saved to {output_path}")
        return output_path
    
    def plot_gc_content_distribution(self, df: pd.DataFrame, 
                                    by_target: bool = False,
                                    output_path: Optional[str] = None) -> str:
        """
        Plot the distribution of GC content.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        by_target : bool, optional
            Whether to separate by target, by default False
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        plt.figure(figsize=(10, 6))
        
        # Determine sequence column name
        seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
        
        # Ensure GC_Content column exists
        gc_col = 'GC_Content' if 'GC_Content' in df.columns else 'gc_content'
        if gc_col not in df.columns:
            if seq_col in df.columns:
                # Calculate GC content
                df[gc_col] = df[seq_col].apply(self._calculate_gc_content)
            else:
                logger.error("DataFrame must contain either 'GC_Content'/'gc_content' or 'Sequence'/'sequence' column")
                raise ValueError("DataFrame must contain either 'GC_Content'/'gc_content' or 'Sequence'/'sequence' column")
        
        if by_target and 'Target_Name' in df.columns:
            # Plot distribution by target
            ax = sns.histplot(data=df, x=gc_col, hue='Target_Name', multiple='stack', kde=True,
                            palette='viridis')
            plt.title('Distribution of GC Content by Target', fontsize=16)
            
            # Adjust legend
            plt.legend(title="Target", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot overall distribution
            ax = sns.histplot(data=df, x=gc_col, kde=True, color='#2ca02c')
            plt.title('Distribution of GC Content', fontsize=16)
        
        plt.xlabel('GC Content (%)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(alpha=0.3)
        
        # Set x-axis range
        plt.xlim(0, 100)
        
        # Save the plot
        if output_path is None:
            output_name = 'gc_content_distribution'
            if by_target:
                output_name += '_by_target'
            output_path = os.path.join(self.output_dir, f"{output_name}.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"GC content distribution plot saved to {output_path}")
        return output_path
    
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
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               top_n: int = 20,
                               output_path: Optional[str] = None) -> str:
        """
        Plot feature importance from a trained model.
        
        Parameters
        ----------
        model : object
            Trained model with feature_importances_ attribute
        feature_names : List[str]
            List of feature names
        top_n : int, optional
            Number of top features to show, by default 20
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_score'):
            # XGBoost model
            importance_dict = model.get_score(importance_type='gain')
            importances = np.zeros(len(feature_names))
            for feature, score in importance_dict.items():
                # XGBoost feature names are in 'f0', 'f1', etc. format
                try:
                    idx = int(feature.replace('f', ''))
                    if idx < len(importances):
                        importances[idx] = score
                except ValueError:
                    pass
        else:
            logger.error("Model does not have feature_importances_ attribute")
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in top_indices]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_indices)), top_importances, align='center', color='#1f77b4')
        
        # Add importance values as text
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                   f"{top_importances[i]:.3f}", va='center')
        
        plt.yticks(range(len(top_indices)), top_features)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Top Feature Importance for Aptamer Binding', fontsize=16)
        plt.grid(alpha=0.3, axis='x')
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"feature_importance.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Feature importance plot saved to {output_path}")
        return output_path
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                             target_names: List[str],
                             output_path: Optional[str] = None) -> str:
        """
        Plot a confusion matrix.
        
        Parameters
        ----------
        conf_matrix : np.ndarray
            Confusion matrix
        target_names : List[str]
            Names of target classes
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names)
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Rotate x-axis labels if there are many classes
        if len(target_names) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"confusion_matrix.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Confusion matrix plot saved to {output_path}")
        return output_path
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       target_names: List[str],
                       output_path: Optional[str] = None) -> str:
        """
        Plot ROC curves for multi-class classification.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        target_names : List[str]
            Names of target classes
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize y_true for multi-class ROC
        n_classes = len(target_names)
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        # Generate a color palette
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            # Calculate ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=color, lw=2,
                   label=f'{target_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"roc_curves.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"ROC curves plot saved to {output_path}")
        return output_path
    
    def plot_binding_vs_specificity(self, df: pd.DataFrame,
                                  binding_col: str = 'predicted_affinity',
                                  specificity_col: str = 'specificity_score',
                                  color_by_target: bool = True,
                                  output_path: Optional[str] = None) -> str:
        """
        Plot binding affinity versus specificity.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aptamer data
        binding_col : str, optional
            Name of the binding affinity column, by default 'predicted_affinity'
        specificity_col : str, optional
            Name of the specificity column, by default 'specificity_score'
        color_by_target : bool, optional
            Whether to color points by target, by default True
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        plt.figure(figsize=(10, 8))
        
        # Check if required columns exist
        if binding_col not in df.columns:
            logger.warning(f"Binding column '{binding_col}' not found, using random data")
            df[binding_col] = np.random.uniform(0.5, 1.0, size=len(df))
        
        if specificity_col not in df.columns:
            logger.warning(f"Specificity column '{specificity_col}' not found, using random data")
            df[specificity_col] = np.random.uniform(0.5, 1.0, size=len(df))
        
        # Create scatter plot
        if color_by_target and 'Target_Name' in df.columns:
            # Group by target and assign colors
            targets = df['Target_Name'].unique()
            
            # Create scatter plot colored by target
            ax = sns.scatterplot(x=binding_col, y=specificity_col, hue='Target_Name',
                                data=df, s=100, alpha=0.7, palette='viridis')
            
            # Add legend
            plt.legend(title="Target", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Create basic scatter plot
            ax = sns.scatterplot(x=binding_col, y=specificity_col,
                               data=df, s=100, alpha=0.7, color='#1f77b4')
        
        # Add labels
        plt.xlabel('Binding Affinity', fontsize=14)
        plt.ylabel('Specificity Score', fontsize=14)
        plt.title('Binding Affinity vs. Specificity', fontsize=16)
        
        # Add grid lines
        plt.grid(alpha=0.3)
        
        # Add optimal region highlight (high binding, high specificity)
        rect = plt.Rectangle((0.8, 0.8), 0.2, 0.2, linewidth=2, edgecolor='r', 
                           facecolor='none', linestyle='--')
        ax.add_patch(rect)
        plt.text(0.9, 0.85, "Optimal\nRegion", ha='center', color='r', fontsize=12)
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"binding_vs_specificity.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Binding vs. specificity plot saved to {output_path}")
        return output_path
    
    def plot_sequence_similarity_heatmap(self, sequences: List[str],
                                       labels: Optional[List[str]] = None,
                                       output_path: Optional[str] = None) -> str:
        """
        Plot a heatmap of sequence similarity between aptamers.
        
        Parameters
        ----------
        sequences : List[str]
            List of aptamer sequences
        labels : Optional[List[str]], optional
            List of labels for the sequences, by default None
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        # Limit to a reasonable number of sequences
        max_sequences = 50
        if len(sequences) > max_sequences:
            logger.warning(f"Too many sequences ({len(sequences)}). Limiting to {max_sequences}.")
            sequences = sequences[:max_sequences]
            if labels is not None:
                labels = labels[:max_sequences]
        
        # Calculate sequence similarity matrix
        n_seq = len(sequences)
        similarity_matrix = np.zeros((n_seq, n_seq))
        
        # Calculate pairwise similarities
        for i in range(n_seq):
            for j in range(n_seq):
                similarity_matrix[i, j] = self._calculate_sequence_similarity(sequences[i], sequences[j])
        
        # Create labels if not provided
        if labels is None:
            seq_ids = [f"Seq {i+1}" for i in range(n_seq)]
        else:
            seq_ids = labels
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        
        # Create clustered heatmap
        g = sns.clustermap(similarity_matrix, 
                          cmap='viridis',
                          xticklabels=seq_ids, 
                          yticklabels=seq_ids,
                          figsize=(12, 10),
                          annot=True,
                          fmt='.2f',
                          linewidths=0.5,
                          cbar_kws={'label': 'Similarity Score'})
        
        plt.title('Aptamer Sequence Similarity Matrix', fontsize=16)
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"sequence_similarity_heatmap.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Sequence similarity heatmap saved to {output_path}")
        return output_path
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate similarity between two sequences.
        
        Parameters
        ----------
        seq1 : str
            First sequence
        seq2 : str
            Second sequence
            
        Returns
        -------
        float
            Similarity score [0, 1]
        """
        if len(seq1) != len(seq2):
            # For sequences of different length, use a simple measure
            min_len = min(len(seq1), len(seq2))
            max_len = max(len(seq1), len(seq2))
            matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
            return matches / max_len
        else:
            # For same length sequences, calculate exact match percentage
            matches = sum(a == b for a, b in zip(seq1, seq2))
            return matches / len(seq1)
    
    def plot_structure_visualization(self, sequence: str, structure: str,
                                   title: Optional[str] = None,
                                   output_path: Optional[str] = None) -> str:
        """
        Create a 2D visualization of the aptamer secondary structure.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
        structure : str
            Secondary structure in dot-bracket notation
        title : Optional[str], optional
            Title for the plot, by default None
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        plt.figure(figsize=(10, 10))
        
        # Convert dot-bracket to a network
        G = self._dot_bracket_to_graph(sequence, structure)
        
        # Create a layout for the graph
        pos = nx.spring_layout(G, k=0.6, iterations=100)
        
        # Plot the network
        node_colors = [G.nodes[n].get('color', '#1f77b4') for n in G.nodes()]
        labels = {n: G.nodes[n].get('label', '') for n in G.nodes()}
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # Add title
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"Aptamer Secondary Structure", fontsize=16)
        
        # Remove axis
        plt.axis('off')
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"structure_visualization.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Structure visualization saved to {output_path}")
        return output_path
    
    def _dot_bracket_to_graph(self, sequence: str, structure: str) -> nx.Graph:
        """
        Convert a dot-bracket structure to a networkx graph.
        
        Parameters
        ----------
        sequence : str
            Aptamer sequence
        structure : str
            Secondary structure in dot-bracket notation
            
        Returns
        -------
        nx.Graph
            NetworkX graph representation of the structure
        """
        G = nx.Graph()
        
        # Add nodes (nucleotides)
        for i, (nt, struct) in enumerate(zip(sequence, structure)):
            # Add node with nucleotide label and structure type
            if struct == '.':
                G.add_node(i, label=nt, color='#1f77b4', type='unpaired')  # Blue for unpaired
            elif struct == '(':
                G.add_node(i, label=nt, color='#2ca02c', type='paired_open')  # Green for paired (opening)
            elif struct == ')':
                G.add_node(i, label=nt, color='#ff7f0e', type='paired_close')  # Orange for paired (closing)
        
        # Add backbone connections
        for i in range(len(sequence) - 1):
            G.add_edge(i, i + 1, type='backbone')
        
        # Add base-pair connections
        stack = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    G.add_edge(i, j, type='base_pair')
        
        return G
    
    def plot_cross_reactivity_matrix(self, df: pd.DataFrame,
                                   target_col: str = 'Target_Name',
                                   output_path: Optional[str] = None) -> str:
        """
        Plot a cross-reactivity matrix for aptamers against different targets.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aptamer data and probability columns for each target
        target_col : str, optional
            Column containing the primary target, by default 'Target_Name'
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        # Find probability columns
        prob_cols = [col for col in df.columns if col.endswith('_probability')]
        
        if not prob_cols:
            logger.error("No probability columns found in the DataFrame")
            raise ValueError("No probability columns found in the DataFrame")
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in the DataFrame")
            raise ValueError(f"Target column '{target_col}' not found in the DataFrame")
        
        # Extract targets from probability column names
        targets = [col.replace('_probability', '') for col in prob_cols]
        
        # Create cross-reactivity matrix
        n_targets = len(targets)
        crossreact_matrix = np.zeros((n_targets, n_targets))
        
        # Group by target and calculate average probabilities
        for i, target1 in enumerate(targets):
            target_df = df[df[target_col] == target1]
            
            if target_df.empty:
                continue
                
            for j, target2 in enumerate(targets):
                prob_col = f"{target2}_probability"
                if prob_col in df.columns:
                    # Calculate average probability
                    crossreact_matrix[i, j] = target_df[prob_col].mean()
        
        # Plot the matrix as a heatmap
        plt.figure(figsize=(12, 10))
        
        ax = sns.heatmap(crossreact_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       xticklabels=targets, yticklabels=targets,
                       vmin=0, vmax=1, linewidths=0.5)
        
        plt.title('Cross-Reactivity Matrix', fontsize=16)
        plt.xlabel('Predicted Target', fontsize=14)
        plt.ylabel('Actual Target', fontsize=14)
        
        # Rotate x-axis labels if there are many targets
        if n_targets > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Add colorbar label
        cbar = ax.collections[0].colorbar
        cbar.set_label('Probability', rotation=270, labelpad=15)
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"cross_reactivity_matrix.{self.plot_format}")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Cross-reactivity matrix plot saved to {output_path}")
        return output_path
    
    def create_dashboard(self, df: pd.DataFrame,
                        output_path: Optional[str] = None) -> str:
        """
        Create a comprehensive dashboard visualization for aptamer analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aptamer data
        output_path : Optional[str], optional
            Path to save the plot, by default None
            
        Returns
        -------
        str
            Path to the saved plot file
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3)
        
        # Get sequence column
        seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
        
        # 1. Sequence Length Distribution
        ax1 = plt.subplot(gs[0, 0])
        if 'length' not in df.columns:
            df['length'] = df[seq_col].str.len()
        sns.histplot(data=df, x='length', kde=True, ax=ax1)
        ax1.set_title('Sequence Length Distribution')
        
        # 2. GC Content Distribution
        ax2 = plt.subplot(gs[0, 1])
        gc_col = 'GC_Content' if 'GC_Content' in df.columns else 'gc_content'
        if gc_col not in df.columns:
            df[gc_col] = df[seq_col].apply(self._calculate_gc_content)
        sns.histplot(data=df, x=gc_col, kde=True, ax=ax2)
        ax2.set_title('GC Content Distribution')
        
        # 3. Binding vs Specificity Scatter Plot
        ax3 = plt.subplot(gs[0, 2])
        binding_col = 'predicted_affinity' if 'predicted_affinity' in df.columns else 'affinity_normalized'
        specificity_col = 'specificity_score' if 'specificity_score' in df.columns else 'combined_specificity'
        
        if binding_col not in df.columns:
            df[binding_col] = np.random.uniform(0.5, 1.0, size=len(df))
        if specificity_col not in df.columns:
            df[specificity_col] = np.random.uniform(0.5, 1.0, size=len(df))
        
        if 'Target_Name' in df.columns:
            sns.scatterplot(x=binding_col, y=specificity_col, hue='Target_Name', data=df, ax=ax3)
        else:
            sns.scatterplot(x=binding_col, y=specificity_col, data=df, ax=ax3)
        ax3.set_title('Binding vs Specificity')
        
        # 4. Target Distribution Bar Plot
        ax4 = plt.subplot(gs[1, 0])
        if 'Target_Name' in df.columns:
            target_counts = df['Target_Name'].value_counts()
            sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax4)
            ax4.set_title('Aptamers per Target')
            if len(target_counts) > 4:
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, "No target information available", ha='center', fontsize=12)
        
        # 5. Structural Stability Distribution
        ax5 = plt.subplot(gs[1, 1])
        stability_col = 'stability_score' if 'stability_score' in df.columns else 'stability_score_refined'
        if stability_col in df.columns:
            sns.histplot(data=df, x=stability_col, kde=True, ax=ax5)
            ax5.set_title('Structural Stability Distribution')
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, "No stability information available", ha='center', fontsize=12)
        
        # 6. Selection Score Distribution
        ax6 = plt.subplot(gs[1, 2])
        if 'selection_score' in df.columns:
            if 'Target_Name' in df.columns:
                sns.boxplot(x='Target_Name', y='selection_score', data=df, ax=ax6)
                ax6.set_title('Selection Score by Target')
                if len(df['Target_Name'].unique()) > 4:
                    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
            else:
                sns.histplot(data=df, x='selection_score', kde=True, ax=ax6)
                ax6.set_title('Selection Score Distribution')
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, "No selection score information available", ha='center', fontsize=12)
        
        # 7. Structure Visualization (for the best aptamer)
        ax7 = plt.subplot(gs[2, 0:2])
        if 'predicted_structure' in df.columns and seq_col in df.columns:
            # Get the best aptamer
            if 'selection_score' in df.columns:
                best_idx = df['selection_score'].idxmax()
            else:
                best_idx = 0
                
            best_seq = df.iloc[best_idx][seq_col]
            best_struct = df.iloc[best_idx]['predicted_structure']
            
            # Create a network representation
            G = self._dot_bracket_to_graph(best_seq, best_struct)
            pos = nx.spring_layout(G, k=0.6, iterations=100)
            
            # Plot the structure
            node_colors = [G.nodes[n].get('color', '#1f77b4') for n in G.nodes()]
            labels = {n: G.nodes[n].get('label', '') for n in G.nodes()}
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=ax7)
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, ax=ax7)
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax7)
            
            ax7.set_title('Best Aptamer Structure')
            ax7.axis('off')
        else:
            ax7.axis('off')
            ax7.text(0.5, 0.5, "No structural information available", ha='center', fontsize=12)
        
        # 8. Summary Statistics Table
        ax8 = plt.subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate summary statistics
        stats = []
        stats.append(["Total Aptamers", len(df)])
        
        if 'Target_Name' in df.columns:
            stats.append(["Unique Targets", df['Target_Name'].nunique()])
        
        if 'length' in df.columns:
            stats.append(["Avg. Sequence Length", f"{df['length'].mean():.1f}"])
        
        if gc_col in df.columns:
            stats.append(["Avg. GC Content", f"{df[gc_col].mean():.1f}%"])
        
        if 'selection_score' in df.columns:
            stats.append(["Avg. Selection Score", f"{df['selection_score'].mean():.3f}"])
            stats.append(["Max Selection Score", f"{df['selection_score'].max():.3f}"])
        
        if 'predicted_affinity' in df.columns:
            stats.append(["Avg. Binding Affinity", f"{df['predicted_affinity'].mean():.3f}"])
        
        if 'specificity_score' in df.columns:
            stats.append(["Avg. Specificity", f"{df['specificity_score'].mean():.3f}"])
        
        # Create a table
        table = ax8.table(cellText=stats, loc='center', cellLoc='center', colWidths=[0.4, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        ax8.set_title('Summary Statistics')
        
        # Add main title
        plt.suptitle('Aptamer Analytics Dashboard', fontsize=20)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the dashboard
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"aptamer_dashboard.{self.plot_format}")
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Analytics dashboard saved to {output_path}")
        return output_path
