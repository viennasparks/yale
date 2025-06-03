import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import pickle
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    """
    Neural network model for binding affinity prediction.
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64, 32],
                dropout_rate: float = 0.3):
        """
        Initialize the neural network model.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        hidden_layers : List[int], optional
            List of hidden layer sizes, by default [128, 64, 32]
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.3
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
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
            Output tensor
        """
        return self.model(x)

class BindingAffinityPredictor:
    """
    Predicts binding affinity between aptamers and target molecules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the binding affinity predictor.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]], optional
            Configuration parameters, by default None
        """
        self.config = config or {}
        self.model_type = self.config.get('model_type', 'xgboost')
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Model hyperparameters
        self.n_estimators = self.config.get('n_estimators', 500)
        self.max_depth = self.config.get('max_depth', 8)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 30)
        
        # Neural network specific parameters
        if self.model_type == 'neural_network':
            self.nn_config = {
                'hidden_layers': self.config.get('hidden_layers', [128, 64, 32]),
                'dropout_rate': self.config.get('dropout_rate', 0.3),
                'batch_size': self.config.get('batch_size', 32),
                'epochs': self.config.get('epochs', 200)
            }
        
        logger.info(f"Initialized BindingAffinityPredictor with model_type={self.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare feature matrix and target values from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        target_column : Optional[str], optional
            Name of the target column, if available, by default None
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (Feature matrix, Target values if target_column is provided)
        """
        # Extract features (excluding target column and non-numeric columns)
        exclude_columns = ['Sequence_ID', 'sequence', 'Sequence', 'Target_Name', 'predicted_structure']
        if target_column:
            exclude_columns.append(target_column)
        
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not feature_columns:
            logger.error("No numeric feature columns found in the DataFrame")
            raise ValueError("No numeric feature columns found in the DataFrame")
        
        self.feature_names = feature_columns
        logger.debug(f"Using {len(feature_columns)} features for binding affinity prediction")
        
        X = df[feature_columns].values
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Extract target values if target column is provided
        y = df[target_column].values if target_column else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, target_column: str, validation_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train a binding affinity prediction model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        target_column : str
            Column containing binding affinity values
        validation_df : Optional[pd.DataFrame], optional
            Validation DataFrame, by default None
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model for binding affinity prediction")
        
        # Prepare features and target
        X, y = self._prepare_features(df, target_column)
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_df is not None:
            X_val, y_val = self._prepare_features(validation_df, target_column)
        
        # Initialize and train appropriate model
        if self.model_type == 'random_forest':
            return self._train_random_forest(X, y, X_val, y_val)
        elif self.model_type == 'gradient_boosting':
            return self._train_gradient_boosting(X, y, X_val, y_val)
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
        Train a Random Forest model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation targets
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics
        """
        logger.debug("Training Random Forest model")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        metrics = {
            "model_type": "random_forest",
            "training_mse": train_mse,
            "training_r2": train_r2,
        }
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                "validation_mse": val_mse,
                "validation_r2": val_r2
            })
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, 
            scoring='neg_mean_squared_error'
        )
        
        metrics.update({
            "cv_mse": -np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        })
        
        logger.info(f"Random Forest training completed - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        return metrics
    
    def _train_gradient_boosting(self, X: np.ndarray, y: np.ndarray, 
                               X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train a Gradient Boosting model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation targets
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics
        """
        logger.debug("Training Gradient Boosting model")
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=0
        )
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        metrics = {
            "model_type": "gradient_boosting",
            "training_mse": train_mse,
            "training_r2": train_r2,
        }
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                "validation_mse": val_mse,
                "validation_r2": val_r2
            })
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, 
            scoring='neg_mean_squared_error'
        )
        
        metrics.update({
            "cv_mse": -np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        })
        
        logger.info(f"Gradient Boosting training completed - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        return metrics
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, 
                     X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train an XGBoost model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation targets
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics
        """
        logger.debug("Training XGBoost model")
        
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Prepare validation set if provided
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
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
            early_stopping_rounds=self.early_stopping_rounds if evals else None,
            verbose_eval=False
        )
        
        # Calculate training metrics
        y_pred = self.model.predict(dtrain)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        metrics = {
            "model_type": "xgboost",
            "training_mse": train_mse,
            "training_r2": train_r2,
            "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        }
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val)
            val_pred = self.model.predict(dval)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                "validation_mse": val_mse,
                "validation_r2": val_r2
            })
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        metrics["feature_importance"] = importance
        
        logger.info(f"XGBoost training completed - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
            
        return metrics
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, 
                            X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Train a neural network model for binding affinity prediction.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation targets
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with training metrics
        """
        logger.debug("Training Neural Network model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get parameters from config
        hidden_layers = self.nn_config.get('hidden_layers', [128, 64, 32])
        dropout_rate = self.nn_config.get('dropout_rate', 0.3)
        batch_size = self.nn_config.get('batch_size', 32)
        epochs = self.nn_config.get('epochs', 200)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        
        # Create validation tensors if provided
        X_val_tensor, y_val_tensor = None, None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = NeuralNetwork(input_dim, hidden_layers, dropout_rate).to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
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
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        metrics = {
            "model_type": "neural_network",
            "training_mse": train_mse,
            "training_r2": train_r2,
            "epochs_completed": min(epoch + 1, epochs),
            "train_losses": train_losses
        }
        
        # Calculate validation metrics
        if X_val_tensor is not None and y_val_tensor is not None:
            with torch.no_grad():
                val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                "validation_mse": val_mse,
                "validation_r2": val_r2,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss
            })
        
        logger.info(f"Neural Network training completed - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict binding affinity for new aptamer sequences.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
            
        Returns
        -------
        np.ndarray
            Predicted binding affinity values
        """
        if self.model is None:
            logger.error("Model has not been trained. Call train() first.")
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        # Prepare features
        X, _ = self._prepare_features(df)
        
        # Make predictions based on model type
        if self.model_type == 'neural_network':
            # Convert to tensor for PyTorch model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy().flatten()
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            y_pred = self.model.predict(dtest)
        else:
            y_pred = self.model.predict(X)
        
        logger.debug(f"Generated binding affinity predictions for {len(df)} sequences")
        return y_pred
    
    def get_top_candidates(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get the top n candidate aptamers with highest predicted binding affinity.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        n : int, optional
            Number of top candidates to return, by default 5
            
        Returns
        -------
        pd.DataFrame
            DataFrame with top candidate aptamers
        """
        # Predict binding affinity
        predictions = self.predict(df)
        
        # Add predictions to DataFrame
        result_df = df.copy()
        result_df['predicted_affinity'] = predictions
        
        # Sort by predicted affinity (higher is better)
        sorted_df = result_df.sort_values('predicted_affinity', ascending=False)
        
        logger.info(f"Selected top {n} candidates based on predicted binding affinity")
        return sorted_df.head(n)
    
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
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'config': self.config
        }
        
        if self.model_type == 'neural_network':
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': {
                    'input_dim': next(self.model.parameters()).shape[1],
                    'hidden_layers': self.nn_config.get('hidden_layers', [128, 64, 32]),
                    'dropout_rate': self.nn_config.get('dropout_rate', 0.3)
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
                self.scaler = model_data['scaler']
                self.config = model_data.get('config', {})
                
                # Extract architecture information
                architecture = checkpoint['model_architecture']
                
                # Recreate the model
                self.model = NeuralNetwork(
                    input_dim=architecture['input_dim'],
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
                self.scaler = model_data['scaler']
                self.config = model_data.get('config', {})
                
                logger.info(f"{self.model_type.capitalize()} model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def evaluate_model(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Evaluate the model on new data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing aptamer data
        target_column : str
            Column containing true binding affinity values
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model has not been trained. Call train() first.")
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        # Prepare features and target
        X, y_true = self._prepare_features(df, target_column)
        
        # Make predictions
        y_pred = self.predict(df)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate median absolute error
        med_ae = np.median(np.abs(y_true - y_pred))
        
        # Calculate explained variance
        from sklearn.metrics import explained_variance_score
        explained_var = explained_variance_score(y_true, y_pred)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "median_absolute_error": med_ae,
            "explained_variance": explained_var,
            "correlation": corr
        }
        
        logger.info(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}, Correlation: {corr:.4f}")
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and their importance scores
        """
        if self.model is None:
            logger.error("Model has not been trained. Call train() first.")
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        if self.model_type == 'neural_network':
            logger.warning("Feature importance not directly available for neural network models")
            return pd.DataFrame({'feature': self.feature_names, 'importance': np.ones(len(self.feature_names))})
        elif self.model_type == 'xgboost':
            # Get feature importance from XGBoost model
            importance_dict = self.model.get_score(importance_type='gain')
            
            # Match feature indices to feature names
            features = []
            scores = []
            
            for feature, score in importance_dict.items():
                # XGBoost feature names are in 'f0', 'f1', etc. format
                idx = int(feature.replace('f', ''))
                if idx < len(self.feature_names):
                    features.append(self.feature_names[idx])
                    scores.append(score)
            
            df = pd.DataFrame({'feature': features, 'importance': scores})
            df = df.sort_values('importance', ascending=False)
            
            return df
        else:
            # For random forest and gradient boosting
            if hasattr(self.model, 'feature_importances_'):
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                })
                df = df.sort_values('importance', ascending=False)
                return df
            else:
                logger.warning(f"Feature importance not available for model type: {self.model_type}")
                return pd.DataFrame({'feature': self.feature_names, 'importance': np.ones(len(self.feature_names))})
