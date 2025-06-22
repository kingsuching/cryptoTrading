import json
import os
import pickle
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm


class XGBoost(BaseEstimator, RegressorMixin):
    """
    Custom implementation of XGBoost algorithm with gradient boosting
    """

    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=1.0, reg_lambda=1.0, reg_alpha=0.0,
                 random_state=None):
        """
        Initialize XGBoost implementation

        Parameters:
        -----------
        base_estimator : sklearn estimator, default=None
            Base model to use for boosting. If None, uses DecisionTreeRegressor
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Step size shrinkage to prevent overfitting
        max_depth : int, default=3
            Maximum depth of base estimators
        subsample : float, default=1.0
            Subsample ratio of training instances
        reg_lambda : float, default=1.0
            L2 regularization term
        reg_alpha : float, default=0.0
            L1 regularization term
        random_state : int, default=None
            Random state for reproducibility
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

        # Initialize containers for models and training history
        self.estimators_ = []
        self.train_scores_ = []
        self.feature_importances_ = None
        if base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )

    def _get_base_estimator(self):
        """Get base estimator with proper configuration"""
        if self.base_estimator is None:
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            # Clone the base estimator
            return clone(self.base_estimator)

    def _compute_gradients(self, y_true, y_pred):
        """
        Compute gradients for gradient boosting
        Using squared loss: L(y, F) = (y - F)^2 / 2
        Gradient: -dL/dF = y - F (residuals)
        """
        return y_true - y_pred

    def _compute_hessians(self, y_true, y_pred):
        """
        Compute second derivatives (Hessians)
        For squared loss: d²L/dF² = 1
        """
        return np.ones_like(y_true)

    def _subsample_data(self, X, gradients, hessians):
        """Apply subsampling to training data"""
        if self.subsample < 1.0:
            n_samples = int(len(X) * self.subsample)
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), n_samples, replace=False)
            return X.iloc[indices], gradients[indices], hessians[indices]
        return X, gradients, hessians

    def fit(self, X, y):
        """
        Fit the XGBoost model

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Training targets

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Convert to numpy arrays for easier manipulation
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Initialize prediction with zeros (or mean)
        y_pred = np.zeros(len(y_array))

        # Store training scores
        self.train_scores_ = []

        # Boosting iterations
        for i in range(self.n_estimators):
            # Compute gradients and hessians
            gradients = self._compute_gradients(y_array, y_pred)
            hessians = self._compute_hessians(y_array, y_pred)

            # Apply subsampling
            X_sub, grad_sub, hess_sub = self._subsample_data(
                X if isinstance(X, pd.DataFrame) else pd.DataFrame(X_array),
                gradients, hessians
            )

            # Fit base estimator on gradients (pseudo-residuals)
            estimator = self._get_base_estimator()
            estimator.fit(X_sub, grad_sub)

            # Make predictions with current estimator
            tree_pred = estimator.predict(X_array)

            # Apply learning rate and update predictions
            y_pred += self.learning_rate * tree_pred

            # Store estimator
            self.estimators_.append(estimator)

            # Calculate and store training score
            mse = mean_squared_error(y_array, y_pred)
            self.train_scores_.append(mse)

            # Early stopping could be implemented here
            if i > 10 and abs(self.train_scores_[-1] - self.train_scores_[-2]) < 1e-6:
                print(f"Early stopping at iteration {i}")
                break

        # Compute feature importances (average from all trees)
        self._compute_feature_importances(X)

        return self

    def _compute_feature_importances(self, X):
        """Compute feature importances from all estimators"""
        if hasattr(self.estimators_[0], 'feature_importances_'):
            n_features = X.shape[1]
            importances = np.zeros(n_features)

            for estimator in self.estimators_:
                importances += estimator.feature_importances_

            self.feature_importances_ = importances / len(self.estimators_)

    def predict(self, X):
        """
        Make predictions using the fitted model

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Features to predict on

        Returns:
        --------
        predictions : numpy.array
            Predicted values
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        predictions = np.zeros(len(X_array))

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X_array)

        return predictions

    def cross_validate(self, X, y, cv=5, return_train_score=False):
        """
        Perform cross-validation on the model

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Training targets
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='neg_mean_squared_error'
            Scoring metric
        return_train_score : bool, default=False
            Whether to return training scores

        Returns:
        --------
        cv_results : dict
            Dictionary containing cross-validation results
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        test_scores = []
        train_scores = [] if return_train_score else None
        splits = kfold.split(X)
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"Fold {fold+1}/{cv}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create and fit model for this fold
            model = XGBoost(
                base_estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_score = mean_squared_error(y_test, y_pred_test)
            test_score **= 0.5
            test_scores.append(test_score)

            if return_train_score:
                y_pred_train = model.predict(X_train)
                train_score = mean_squared_error(y_train, y_pred_train)
                train_score **= 0.5
                train_scores.append(train_score)

        cv_results = {
            'test_scores': np.array(test_scores),
            'test_score_mean': np.mean(test_scores),
            'test_score_std': np.std(test_scores)
        }

        if return_train_score:
            cv_results.update({
                'train_scores': np.array(train_scores),
                'train_score_mean': np.mean(train_scores),
                'train_score_std': np.std(train_scores)
            })

        return cv_results

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'reg_lambda': self.reg_lambda,
            'reg_alpha': self.reg_alpha,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def tune_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='rmse',
                             verbose=True, n_jobs=1):
        """
        Perform grid search cross-validation for hyperparameter tuning

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Training targets
        param_grid : dict, default=None
            Dictionary with parameters names as keys and lists of parameter settings.
            If None, uses a default parameter grid.
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='rmse'
            Scoring metric ('rmse' or 'mse')
        verbose : bool, default=True
            Whether to print progress
        n_jobs : int, default=1
            Number of parallel jobs (placeholder for future implementation)

        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        from sklearn.model_selection import ParameterGrid
        import time

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'reg_lambda': [0.1, 1.0, 2.0]
            }

        best_score = float('inf')  # We want to minimize RMSE/MSE
        best_params = None
        all_results = []

        # Iterate through all parameter combinations
        for i, params in enumerate(tqdm(ParameterGrid(param_grid))):

            # Create model with current parameters
            model = XGBoost(**params, random_state=self.random_state)

            # Perform cross-validation
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            fold_scores = []

            for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate score
                if scoring == 'rmse':
                    score = mean_squared_error(y_test, y_pred) ** 0.5
                else:  # mse
                    score = mean_squared_error(y_test, y_pred)

                fold_scores.append(score)

            # Calculate mean score for this parameter combination
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            # Store results
            result = {
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'individual_scores': fold_scores.copy()
            }
            all_results.append(result)

            # Update best parameters
            if mean_score < best_score:
                best_score = mean_score
                best_params = params.copy()

        # Sort results by score
        all_results.sort(key=lambda x: x['mean_score'])

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.tuning_results_ = all_results
        return best_params

    def get_tuning_results(self):
        """
        Get detailed tuning results as a pandas DataFrame

        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame containing all parameter combinations and their scores
        """
        if not self.tuning_results_:
            print("No tuning results available. Run tune_hyperparameters() first.")
            return None

        # Flatten the results for DataFrame creation
        flattened_results = []
        for result in self.tuning_results_:
            flat_result = result['params'].copy()
            flat_result['mean_score'] = result['mean_score']
            flat_result['std_score'] = result['std_score']
            flattened_results.append(flat_result)

        results_df = pd.DataFrame(flattened_results)
        results_df = results_df.sort_values('mean_score').reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)

        return results_df

    def fit_best(self, X, y):
        """
        Fit the model using the best parameters found during tuning

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series
            Training targets

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        if self.best_params_ is None:
            raise ValueError("No best parameters available. Run tune_hyperparameters() first.")

        # Update parameters with best found parameters
        for param, value in self.best_params_.items():
            setattr(self, param, value)

        # Fit with best parameters
        return self.fit(X, y)

    def save_model(self, filepath, method='joblib'):
        """
        Save the entire model to a file

        Parameters:
        -----------
        filepath : str
            Path where to save the model
        method : str, default='joblib'
            Method to use for saving ('joblib', 'pickle')
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        if method == 'joblib':
            joblib.dump(self, filepath)
        elif method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Method must be 'joblib' or 'pickle'")

        print(f"Model saved to {filepath} using {method}")

    @classmethod
    def load_model(cls, filepath, method='joblib'):
        """
        Load a model from a file

        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        method : str, default='joblib'
            Method used for loading ('joblib', 'pickle')

        Returns:
        --------
        model : XGBoost
            Loaded XGBoost model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        if method == 'joblib':
            model = joblib.load(filepath)
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError("Method must be 'joblib' or 'pickle'")

        print(f"Model loaded from {filepath} using {method}")
        return model

    def state_dict(self):
        """
        Get the state dictionary of the model (similar to PyTorch's state_dict)

        Returns:
        --------
        state : dict
            Dictionary containing all model parameters and trained components
        """
        state = {
            # Hyperparameters
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'reg_lambda': self.reg_lambda,
                'reg_alpha': self.reg_alpha,
                'random_state': self.random_state
            },

            # Trained components
            'estimators': self.estimators_,
            'train_scores': self.train_scores_,
            'feature_importances': self.feature_importances_,

            # Base estimator info
            'base_estimator_class': self.base_estimator.__class__.__name__ if self.base_estimator else None,
            'base_estimator_params': self.base_estimator.get_params() if self.base_estimator else None,

            # Tuning results (if available)
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'tuning_results': self.tuning_results_,

            # Metadata
            'model_version': '1.0',
            'created_timestamp': datetime.now().isoformat(),
            'is_fitted': len(self.estimators_) > 0
        }

        return state

    def load_state_dict(self, state_dict):
        """
        Load model state from a state dictionary

        Parameters:
        -----------
        state_dict : dict
            Dictionary containing model state (from state_dict() method)
        """
        # Load hyperparameters
        hyperparams = state_dict.get('hyperparameters', {})
        for key, value in hyperparams.items():
            setattr(self, key, value)

        # Load trained components
        self.estimators_ = state_dict.get('estimators', [])
        self.train_scores_ = state_dict.get('train_scores', [])
        self.feature_importances_ = state_dict.get('feature_importances')

        # Load tuning results
        self.best_params_ = state_dict.get('best_params')
        self.best_score_ = state_dict.get('best_score')
        self.tuning_results_ = state_dict.get('tuning_results', [])

        # Reconstruct base estimator
        base_estimator_class = state_dict.get('base_estimator_class')
        base_estimator_params = state_dict.get('base_estimator_params')

        if base_estimator_class == 'DecisionTreeRegressor':
            self.base_estimator = DecisionTreeRegressor(**base_estimator_params)
        elif base_estimator_class is None:
            self.base_estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )

        print(f"Model state loaded successfully. Fitted: {state_dict.get('is_fitted', False)}")

    def save_state_dict(self, filepath):
        """
        Save the model state dictionary to a file

        Parameters:
        -----------
        filepath : str
            Path where to save the state dictionary
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        state = self.state_dict()

        # Use joblib for the state dict as it handles sklearn objects better
        if filepath.endswith('.json'):
            # For json, we need to handle non-serializable objects
            json_state = self._prepare_state_for_json(state)
            with open(filepath, 'w') as f:
                json.dump(json_state, f, indent=2)
        else:
            # Use joblib for full state preservation
            joblib.dump(state, filepath)

        print(f"State dictionary saved to {filepath}")

    def _prepare_state_for_json(self, state):
        """
        Prepare state dictionary for JSON serialization by handling non-serializable objects
        """
        json_state = state.copy()

        # Remove non-serializable objects for JSON
        json_state['estimators'] = f"<{len(state['estimators'])} trained estimators>"
        json_state['feature_importances'] = state['feature_importances'].tolist() if state[
                                                                                         'feature_importances'] is not None else None
        json_state['train_scores'] = [float(score) for score in state['train_scores']]

        return json_state

    @classmethod
    def load_state_dict_from_file(cls, filepath):
        """
        Create a new model instance and load state from a file

        Parameters:
        -----------
        filepath : str
            Path to the saved state dictionary file

        Returns:
        --------
        model : XGBoost
            New XGBoost model with loaded state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State dictionary file not found: {filepath}")

        # Create new instance
        model = cls()

        # Load state
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            print(
                "Warning: JSON format doesn't preserve trained estimators. Use joblib format for complete model saving.")
        else:
            state_dict = joblib.load(filepath)

        model.load_state_dict(state_dict)

        return model

    def model_info(self):
        """
        Get information about the current model state

        Returns:
        --------
        info : dict
            Dictionary containing model information
        """
        info = {
            'is_fitted': len(self.estimators_) > 0,
            'n_estimators_fitted': len(self.estimators_),
            'n_estimators_configured': self.n_estimators,
            'has_feature_importances': self.feature_importances_ is not None,
            'has_tuning_results': len(self.tuning_results_) > 0,
            'best_score': self.best_score_,
            'hyperparameters': self.get_params()
        }

        if self.train_scores_:
            info['final_training_mse'] = self.train_scores_[-1]
            info['initial_training_mse'] = self.train_scores_[0]

        return info