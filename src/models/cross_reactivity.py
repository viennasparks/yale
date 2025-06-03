import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import pickle
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class CrossReactivityClassifier(nn.Module):
    """
    Neural network model for cross-reactivity classification.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_layers: List[int] = [256, 128, 64],
               dropout_rate: float = 0.4):
        """
        Initialize the neural network classifier.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        num_classes : int
            Number of target classes
        hidden_layers : List[int], optional
            Sizes of hidden layers, by default [256, 128, 64]
        dropout_rate : float, optional
            Dropout rate, by default 0.4
        """
        super(CrossReactivityClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Class logits
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """
        Extract features from the input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Extracted features
        """
        return self.feature_extractor(x)

class CrossReactivityAnalyzer:
    """
    Analyze and predict cross-reactivity between aptamers and non-target molecules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CrossReactivityAnalyzer.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        
        self.model_type = self.config.get('model_type', 'xgboost')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        
        # Model hyperparameters
        self.n_estimators = self.config.get('n_estimators', 300)
        self.max_depth = self.config.get('max_depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.05)
        
        # Neural network specific parameters
        if self.model_type == 'neural_network':
            self.nn_config = {
                'hidden_layers': self.config.get('hidden_layers', [256, 128, 64]),
                'dropout_rate': self.config.get('dropout_rate', 0.4),
                'batch_size': self.config.get('batch_size', 32),
                'epochs': self.config.get('epochs', 100)
            }
        
        logger.info(f"Initialized CrossReactivityAnalyzer with model_type={self.model_type}")
    
    def prepare_cross_reactivity_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for cross-reactivity analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data with multiple targets
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Feature matrix, Target labels)
        """
        # Ensure Target_Name column exists
        if 'Target_Name' not in df.columns:
            logger.error("DataFrame must contain a 'Target_Name' column")
            raise ValueError("DataFrame must contain a 'Target_Name' column")
        
        # Exclude non-numeric and identifier columns
        exclude_columns = ['Sequence_ID', 'sequence', 'Sequence', 'Target_Name', 'predicted_structure']
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not feature_columns:
            logger.error("No numeric feature columns found in the DataFrame")
            raise ValueError("No numeric feature columns found in the DataFrame")
        
        self.feature_names = feature_columns
        
        # Extract features and scale them
        X = df[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target names
        self.target_names = df['Target_Name'].unique()
        logger.info(f"Found {len(self.target_names)} unique targets: {self.target_names}")
        
        y = self.label_encoder.fit_transform(df['Target_Name'])
        
        return X_scaled, y
    
    def train_cross_reactivity_model(self, df: pd.DataFrame, 
                                    validation_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train a model to predict cross-reactivity.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data with multiple targets
        validation_df : Optional[pd.DataFrame], optional
            Validation DataFrame, by default None
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model for cross-reactivity analysis")
        
        # Prepare training data
        X, y = self.prepare_cross_reactivity_data(df)
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_df is not None:
            X_val, y_val = self.prepare_cross_reactivity_data(validation_df)
        
        # Initialize and train the appropriate model
        if self.model_type == 'random_forest':
            return self._train_random_forest(X, y, X_val, y_val)
        elif self.model_type == 'xgboost':
            return self._train_xgboost(X, y, X_val, y_val)
        elif self.model_type == 'neural_network':
            return self._train_neural_network(X, y, X_val, y_val)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, 
                            X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train a Random Forest model for cross-reactivity prediction.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation labels
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.debug("Training Random Forest model")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        train_accuracy = accuracy_score(y, y_pred)
        train_conf_matrix = confusion_matrix(y, y_pred)
        train_class_report = classification_report(y, y_pred, output_dict=True)
        
        metrics = {
            "model_type": "random_forest",
            "training_accuracy": train_accuracy,
            "training_confusion_matrix": train_conf_matrix,
            "training_classification_report": train_class_report
        }
        
        # Calculate ROC AUC for multiclass
        train_roc_auc = self._calculate_multiclass_roc_auc(y, y_prob)
        metrics["training_roc_auc"] = train_roc_auc
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_prob = self.model.predict_proba(X_val)
            
            val_accuracy = accuracy_score(y_val, val_pred)
            val_conf_matrix = confusion_matrix(y_val, val_pred)
            val_class_report = classification_report(y_val, val_pred, output_dict=True)
            val_roc_auc = self._calculate_multiclass_roc_auc(y_val, val_prob)
            
            metrics.update({
                "validation_accuracy": val_accuracy,
                "validation_confusion_matrix": val_conf_matrix,
                "validation_classification_report": val_class_report,
                "validation_roc_auc": val_roc_auc
            })
        
        logger.info(f"Random Forest training completed - Accuracy: {train_accuracy:.4f}")
        return metrics
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, 
                     X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train an XGBoost model for cross-reactivity prediction.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation labels
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.debug("Training XGBoost model")
        
        num_classes = len(np.unique(y))
        
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Prepare validation set if provided
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
        
        # Set parameters
        params = {
            'objective': 'multi:softprob',
            'eval_metric': ['mlogloss', 'merror'],
            'num_class': num_classes,
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train the model
        self.model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=20 if evals else None,
            verbose_eval=False
        )
        
        # Calculate training metrics
        y_prob = self.model.predict(dtrain)
        y_pred = np.argmax(y_prob, axis=1)
        
        train_accuracy = accuracy_score(y, y_pred)
        train_conf_matrix = confusion_matrix(y, y_pred)
        train_class_report = classification_report(y, y_pred, output_dict=True)
        train_roc_auc = self._calculate_multiclass_roc_auc(y, y_prob)
        
        metrics = {
            "model_type": "xgboost",
            "training_accuracy": train_accuracy,
            "training_confusion_matrix": train_conf_matrix,
            "training_classification_report": train_class_report,
            "training_roc_auc": train_roc_auc,
            "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        }
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val)
            val_prob = self.model.predict(dval)
            val_pred = np.argmax(val_prob, axis=1)
            
            val_accuracy = accuracy_score(y_val, val_pred)
            val_conf_matrix = confusion_matrix(y_val, val_pred)
            val_class_report = classification_report(y_val, val_pred, output_dict=True)
            val_roc_auc = self._calculate_multiclass_roc_auc(y_val, val_prob)
            
            metrics.update({
                "validation_accuracy": val_accuracy,
                "validation_confusion_matrix": val_conf_matrix,
                "validation_classification_report": val_class_report,
                "validation_roc_auc": val_roc_auc
            })
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        metrics["feature_importance"] = importance
        
        logger.info(f"XGBoost training completed - Accuracy: {train_accuracy:.4f}")
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
            
        return metrics
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, 
                            X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train a neural network model for cross-reactivity prediction.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation labels
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.debug("Training Neural Network model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get parameters from config
        hidden_layers = self.nn_config.get('hidden_layers', [256, 128, 64])
        dropout_rate = self.nn_config.get('dropout_rate', 0.4)
        batch_size = self.nn_config.get('batch_size', 32)
        epochs = self.nn_config.get('epochs', 100)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        
        # Create validation tensors if provided
        X_val_tensor, y_val_tensor = None, None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = CrossReactivityClassifier(input_dim, num_classes, hidden_layers, dropout_rate).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting Neural Network training for {epochs} epochs")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Save predictions for metrics
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate training metrics
            train_accuracy = accuracy_score(all_labels, all_preds)
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
                    # Calculate validation accuracy
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_predicted.cpu().numpy())
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save best model
                        best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= 10:  # Early stopping patience
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break
                
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                                f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, "
                                f"Val Acc: {val_accuracy:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                                f"Train Acc: {train_accuracy:.4f}")
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(X_tensor)
            _, train_preds = torch.max(train_outputs, 1)
            train_probs = torch.softmax(train_outputs, dim=1).cpu().numpy()
        
        train_preds = train_preds.cpu().numpy()
        train_accuracy = accuracy_score(y, train_preds)
        train_conf_matrix = confusion_matrix(y, train_preds)
        train_class_report = classification_report(y, train_preds, output_dict=True)
        train_roc_auc = self._calculate_multiclass_roc_auc(y, train_probs)
        
        metrics = {
            "model_type": "neural_network",
            "training_accuracy": train_accuracy,
            "training_confusion_matrix": train_conf_matrix,
            "training_classification_report": train_class_report,
            "training_roc_auc": train_roc_auc,
            "epochs_completed": min(epoch + 1, epochs),
            "train_losses": train_losses
        }
        
        # Calculate validation metrics
        if X_val_tensor is not None and y_val_tensor is not None:
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                _, val_preds = torch.max(val_outputs, 1)
                val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
            
            val_preds = val_preds.cpu().numpy()
            val_accuracy = accuracy_score(y_val, val_preds)
            val_conf_matrix = confusion_matrix(y_val, val_preds)
            val_class_report = classification_report(y_val, val_preds, output_dict=True)
            val_roc_auc = self._calculate_multiclass_roc_auc(y_val, val_probs)
            
            metrics.update({
                "validation_accuracy": val_accuracy,
                "validation_confusion_matrix": val_conf_matrix,
                "validation_classification_report": val_class_report,
                "validation_roc_auc": val_roc_auc,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss
            })
        
        logger.info(f"Neural Network training completed - Accuracy: {train_accuracy:.4f}")
        return metrics
    
    def _calculate_multiclass_roc_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate ROC AUC for multiclass classification using one-vs-rest approach.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping class labels to ROC AUC scores
        """
        roc_auc = {}
        
        # Handle different shapes of y_prob
        if len(y_prob.shape) == 1:
            # Binary classification case
            roc_auc["average"] = roc_auc_score(y_true, y_prob)
        else:
            # Multiclass case
            for i, target_name in enumerate(self.label_encoder.classes_):
                # One-vs-Rest approach for multiclass ROC
                y_true_binary = (y_true == i).astype(int)
                
                try:
                    # Ensure there are both positive and negative samples
                    if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true_binary):
                        roc_auc[target_name] = roc_auc_score(y_true_binary, y_prob[:, i])
                    else:
                        roc_auc[target_name] = float('nan')
                except Exception as e:
                    logger.warning(f"Error calculating ROC AUC for class {target_name}: {str(e)}")
                    roc_auc[target_name] = float('nan')
            
            # Calculate macro average
            valid_scores = [score for score in roc_auc.values() if not np.isnan(score)]
            roc_auc["average"] = np.mean(valid_scores) if valid_scores else float('nan')
        
        return roc_auc
    
    def predict_cross_reactivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cross-reactivity probabilities for new aptamer sequences.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with cross-reactivity predictions
        """
        if self.model is None:
            logger.error("Model has not been trained. Call train_cross_reactivity_model() first.")
            raise RuntimeError("Model has not been trained. Call train_cross_reactivity_model() first.")
        
        # Extract and scale features
        exclude_columns = ['Sequence_ID', 'sequence', 'Sequence', 'Target_Name', 'predicted_structure']
        feature_columns = [col for col in self.feature_names if col in df.columns]
        
        if not feature_columns or len(feature_columns) != len(self.feature_names):
            missing = set(self.feature_names) - set(feature_columns)
            logger.error(f"Missing required feature columns: {missing}")
            raise ValueError(f"Missing required feature columns: {missing}")
        
        X = df[feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions based on model type
        if self.model_type == 'neural_network':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Get predicted class
            preds = np.argmax(probs, axis=1)
            
            # Convert predicted indices back to original target names
            pred_targets = self.label_encoder.inverse_transform(preds)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X_scaled)
            probs = self.model.predict(dtest)
            preds = np.argmax(probs, axis=1)
            pred_targets = self.label_encoder.inverse_transform(preds)
        else:
            # For sklearn models
            probs = self.model.predict_proba(X_scaled)
            preds = self.model.predict(X_scaled)
            pred_targets = self.label_encoder.inverse_transform(preds)
        
        # Create results DataFrame
        results = df.copy()
        
        # Add prediction probabilities for each target
        for i, target in enumerate(self.label_encoder.classes_):
            results[f'{target}_probability'] = probs[:, i]
        
        # Add predicted target
        results['predicted_target'] = pred_targets
        logger.debug(f"Generated cross-reactivity predictions for {len(df)} sequences")
        
        return results
    
    def identify_cross_reactive_aptamers(self, df: pd.DataFrame, 
                                        threshold: float = 0.3) -> pd.DataFrame:
        """
        Identify aptamers that may be cross-reactive based on prediction probabilities.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cross-reactivity predictions
        threshold : float, optional
            Threshold for considering a non-primary target as cross-reactive, by default 0.3
            
        Returns
        -------
        pd.DataFrame
            DataFrame with cross-reactive aptamers
        """
        # Get prediction results if not already present
        if 'predicted_target' not in df.columns:
            df = self.predict_cross_reactivity(df)
        
        # Create a copy for results
        results = df.copy()
        
        # Identify primary target for each aptamer
        target_cols = [col for col in df.columns if col.endswith('_probability')]
        
        # Add cross-reactivity flag
        results['is_cross_reactive'] = False
        results['cross_reactive_targets'] = ''
        results['specificity_score'] = 0.0
        
        for idx, row in results.iterrows():
            # Get the highest probability target
            primary_target = row['predicted_target']
            
            # Check for other targets with high probability
            cross_reactive_targets = []
            
            for col in target_cols:
                target = col.replace('_probability', '')
                if target != primary_target and row[col] >= threshold:
                    cross_reactive_targets.append(target)
            
            if cross_reactive_targets:
                results.at[idx, 'is_cross_reactive'] = True
                results.at[idx, 'cross_reactive_targets'] = ', '.join(cross_reactive_targets)
            
            # Calculate specificity score
            probs = [row[col] for col in target_cols]
            sorted_probs = sorted(probs, reverse=True)
            
            # Specificity score: difference between highest and second highest probability
            if len(sorted_probs) >= 2:
                results.at[idx, 'specificity_score'] = sorted_probs[0] - sorted_probs[1]
            else:
                results.at[idx, 'specificity_score'] = 1.0
        
        logger.info(f"Identified {results['is_cross_reactive'].sum()} cross-reactive aptamers "
                   f"(threshold={threshold})")
        
        return results
    
    def visualize_cross_reactivity(self, df: pd.DataFrame, 
                                  method: str = 'tsne',
                                  output_path: Optional[str] = None,
                                  show_plot: bool = True) -> None:
        """
        Visualize cross-reactivity between different targets using dimensionality reduction.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with aptamer data
        method : str, optional
            Dimensionality reduction method ('pca' or 'tsne'), by default 'tsne'
        output_path : Optional[str], optional
            Path to save the visualization, by default None
        show_plot : bool, optional
            Whether to display the plot, by default True
        """
        if self.model is None:
            logger.warning("Model has not been trained, using raw features for visualization")
        
        # Prepare data
        X, y = self.prepare_cross_reactivity_data(df)
        
        # Use model's feature extractor if available (for neural network)
        if self.model_type == 'neural_network' and self.model is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            self.model.eval()
            with torch.no_grad():
                X_features = self.model.get_features(X_tensor).cpu().numpy()
        else:
            # Use raw features
            X_features = X
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_features)-1))
            X_reduced = reducer.fit_transform(X_features)
            title = 'Cross-Reactivity Visualization using t-SNE'
        else:  # default to PCA
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_features)
            title = f'Cross-Reactivity Visualization using PCA\n' \
                   f'(explained variance: PC1={reducer.explained_variance_ratio_[0]:.2%}, ' \
                   f'PC2={reducer.explained_variance_ratio_[1]:.2%})'
        
        # Set up plot
        plt.figure(figsize=(12, 10))
        
        # Get target names
        target_names = self.label_encoder.classes_
        
        # Plot each target class
        for i, target in enumerate(target_names):
            mask = (y == i)
            if np.any(mask):  # Only plot if there are examples for this target
                plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                           label=target, alpha=0.7, s=100, marker=f"${i}$")
        
        plt.title(title, fontsize=16)
        plt.xlabel('Component 1', fontsize=14)
        plt.ylabel('Component 2', fontsize=14)
        plt.legend(fontsize=12, markerscale=1.5)
        plt.grid(alpha=0.3)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cross-reactivity visualization saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def calculate_specificity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a specificity score for each aptamer based on prediction probabilities.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cross-reactivity predictions
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added specificity scores
        """
        # Get prediction results if not already present
        if not any(col.endswith('_probability') for col in df.columns):
            df = self.predict_cross_reactivity(df)
        
        # Create a copy for results
        results = df.copy()
        
        # Get probability columns
        prob_cols = [col for col in df.columns if col.endswith('_probability')]
        
        # Calculate specificity score
        # Higher score means higher specificity (less cross-reactivity)
        def calculate_score(row):
            probs = [row[col] for col in prob_cols]
            # Sort probabilities in descending order
            sorted_probs = sorted(probs, reverse=True)
            
            # Calculate the difference between the highest and second highest probability
            if len(sorted_probs) >= 2:
                return sorted_probs[0] - sorted_probs[1]
            else:
                return 1.0  # If only one target, specificity is perfect
        
        results['specificity_score'] = results.apply(calculate_score, axis=1)
        
        # Also add entropy-based score (lower entropy means higher specificity)
        def calculate_entropy(row):
            probs = np.array([row[col] for col in prob_cols])
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            probs = probs / probs.sum()  # Normalize
            return -np.sum(probs * np.log2(probs))
        
        results['entropy'] = results.apply(calculate_entropy, axis=1)
        # Normalize entropy to [0, 1] with 1 being most specific
        max_entropy = np.log2(len(prob_cols))
        results['entropy_specificity'] = 1 - (results['entropy'] / max_entropy)
        
        # Combined score (average of both metrics)
        results['combined_specificity'] = (results['specificity_score'] + results['entropy_specificity']) / 2
        
        logger.debug(f"Calculated specificity scores for {len(df)} sequences")
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            logger.error("No trained model to save.")
            raise RuntimeError("No trained model to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'target_names': self.target_names,
            'scaler': self.scaler,
            'config': self.config
        }
        
        if self.model_type == 'neural_network':
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': {
                    'input_dim': next(self.model.parameters()).shape[1],
                    'num_classes': len(self.target_names),
                    'hidden_layers': self.nn_config.get('hidden_layers', [256, 128, 64]),
                    'dropout_rate': self.nn_config.get('dropout_rate', 0.4)
                },
                'model_data': model_data
            }, filepath)
            logger.info(f"Neural network model saved to {filepath}")
        else:
            # Save scikit-learn or XGBoost model
            model_data['model'] = self.model
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"{self.model_type.capitalize()} model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            if filepath.endswith('.pt'):
                # Load PyTorch model
                checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
                model_data = checkpoint['model_data']
                
                # Extract model information
                self.model_type = model_data['model_type']
                self.feature_names = model_data['feature_names']
                self.label_encoder = model_data['label_encoder']
                self.target_names = model_data['target_names']
                self.scaler = model_data['scaler']
                self.config = model_data.get('config', {})
                
                # Extract architecture information
                architecture = checkpoint['model_architecture']
                
                # Recreate the model
                self.model = CrossReactivityClassifier(
                    input_dim=architecture['input_dim'],
                    num_classes=architecture['num_classes'],
                    hidden_layers=architecture['hidden_layers'],
                    dropout_rate=architecture['dropout_rate']
                )
                
                # Load the weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info(f"Neural network model loaded from {filepath}")
            else:
                # Load scikit-learn or XGBoost model
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model_type = model_data['model_type']
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.label_encoder = model_data['label_encoder']
                self.target_names = model_data['target_names']
                self.scaler = model_data['scaler']
                self.config = model_data.get('config', {})
                
                logger.info(f"{self.model_type.capitalize()} model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise RuntimeError(f"Error loading model: {str(e)}")
