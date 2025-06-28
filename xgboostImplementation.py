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

from CONSTANTS import TEST_DAYS


class XGBoost(BaseEstimator, RegressorMixin):
    """
    Custom XGBoost implementation specifically for Bitcoin price prediction
    Outputs a 1x7 list of Bitcoin close prices (self.output_size-day forecast)
    """

    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=1.0, reg_lambda=1.0, reg_alpha=0.0,
                 random_state=None):
        """
        Initialize XGBoost Bitcoin Predictor

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
        self.output_size = TEST_DAYS  # Fixed to self.output_size days for Bitcoin price prediction
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
        self.best_params_ = None
        self.best_score_ = None
        self.tuning_results_ = []

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

    def _calculate_rmse_nday(self, y_true, y_pred):
        """
        Calculate RMSE for self.output_size-day predictions where both y_true and y_pred are lists of self.output_size numbers
        Formula: sqrt(sum((y_true[i] - y_pred[i])^2) / self.output_size)

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, self.output_size) or (self.output_size,)
            True Bitcoin prices for self.output_size days
        y_pred : array-like, shape (n_samples, self.output_size) or (self.output_size,)
            Predicted Bitcoin prices for self.output_size days

        Returns:
        --------
        rmse : float or array
            Root Mean Squared Error for self.output_size-day predictions
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle single sample case
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        # Ensure both have self.output_size columns
        if y_true.shape[1] != self.output_size or y_pred.shape[1] != self.output_size:
            raise ValueError(f"Both y_true and y_pred must have {self.output_size} columns. Got {y_true.shape[1]} and {y_pred.shape[1]}")

        # Calculate RMSE for each sample
        squared_errors = (y_true - y_pred) ** 2
        mse = np.mean(squared_errors, axis=1)  # Mean across self.output_size days for each sample
        rmse = np.sqrt(mse)

        # Return single value if single sample, otherwise return array
        return rmse[0] if len(rmse) == 1 else rmse

    def fit(self, X, y):
        """
        Fit the XGBoost model for Bitcoin price prediction

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features (technical indicators, price history, etc.)
        y : pandas.Series or pandas.DataFrame
            Training targets (Bitcoin close prices)
            If DataFrame, should have self.output_size columns for self.output_size-day predictions

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Convert to numpy arrays for easier manipulation
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Handle multi-output targets (self.output_size-day predictions)
        if isinstance(y, pd.DataFrame):
            y_array = y.values
        elif isinstance(y, pd.Series):
            y_array = y.values.reshape(-1, 1)
        else:
            y_array = np.array(y)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)

        # Ensure we have self.output_size output columns
        if y_array.shape[1] != self.output_size:
            # If single target, replicate for self.output_size days (simple approach)
            if y_array.shape[1] == 1:
                y_array = np.repeat(y_array, self.output_size, axis=1)
            else:
                raise ValueError(f"Target should have {self.output_size} columns for {self.output_size}-day prediction, got {y_array.shape[1]}")

        # Initialize predictions with zeros
        y_pred = np.zeros_like(y_array)

        # Store training scores
        self.train_scores_ = []
        self.estimators_ = []

        # Train separate models for each day (multi-output approach)
        for day in range(self.output_size):
            day_estimators = []
            y_day = y_array[:, day]
            y_pred_day = np.zeros(len(y_day))

            # Boosting iterations for this day
            for i in range(self.n_estimators):
                # Compute gradients and hessians
                gradients = self._compute_gradients(y_day, y_pred_day)
                hessians = self._compute_hessians(y_day, y_pred_day)

                # Apply subsampling
                X_sub, grad_sub, hess_sub = self._subsample_data(
                    X if isinstance(X, pd.DataFrame) else pd.DataFrame(X_array),
                    gradients, hessians
                )

                # Fit base estimator on gradients
                estimator = self._get_base_estimator()
                estimator.fit(X_sub, grad_sub)

                # Make predictions with current estimator
                tree_pred = estimator.predict(X_array)

                # Apply learning rate and update predictions
                y_pred_day += self.learning_rate * tree_pred
                day_estimators.append(estimator)

                # Early stopping check
                if i > 10:
                    current_mse = mean_squared_error(y_day, y_pred_day)
                    if i == 11:
                        prev_mse = current_mse
                    elif abs(current_mse - prev_mse) < 1e-6:
                        break
                    prev_mse = current_mse

            self.estimators_.append(day_estimators)
            y_pred[:, day] = y_pred_day

        # Calculate overall training score using self.output_size-day RMSE
        overall_rmse = self._calculate_rmse_nday(y_array, y_pred)
        self.train_scores_.append(np.mean(overall_rmse) if isinstance(overall_rmse, np.ndarray) else overall_rmse)

        # Compute feature importances
        self._compute_feature_importances(X)

        return self

    def _compute_feature_importances(self, X):
        """Compute feature importances from all estimators"""
        if len(self.estimators_) > 0 and len(self.estimators_[0]) > 0:
            if hasattr(self.estimators_[0][0], 'feature_importances_'):
                n_features = X.shape[1]
                importances = np.zeros(n_features)
                total_estimators = 0

                for day_estimators in self.estimators_:
                    for estimator in day_estimators:
                        importances += estimator.feature_importances_
                        total_estimators += 1

                self.feature_importances_ = importances / total_estimators

    def predict(self, X):
        """
        Make Bitcoin price predictions for the next self.output_size days

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Features to predict on

        Returns:
        --------
        predictions : numpy.array or list
            1x7 array/list of Bitcoin close prices for the next self.output_size days
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Handle single sample prediction
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        predictions = np.zeros((X_array.shape[0], self.output_size))

        # Make predictions for each day
        for day in range(self.output_size):
            day_predictions = np.zeros(X_array.shape[0])

            for estimator in self.estimators_[day]:
                day_predictions += self.learning_rate * estimator.predict(X_array)

            predictions[:, day] = day_predictions

        # Return as 1x7 list for single prediction, or full array for multiple
        if predictions.shape[0] == 1:
            return predictions[0].tolist()  # Return as list of self.output_size numbers
        else:
            return predictions

    def predict_single(self, X):
        """
        Convenience method to ensure single prediction returns 1x7 list

        Parameters:
        -----------
        X : pandas.DataFrame, pandas.Series, or numpy.array
            Single sample features

        Returns:
        --------
        prediction : list
            List of self.output_size Bitcoin close price predictions
        """
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
            if X.shape[0] > 1:
                X = X[:1]  # Take only first row
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            elif X.shape[0] > 1:
                X = X[:1]

        prediction = self.predict(X)
        return prediction if isinstance(prediction, list) else prediction.tolist()

    def cross_validate(self, X, y, cv=5, return_train_score=False, verbose=True):
        """
        Perform cross-validation on the Bitcoin price prediction model

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.DataFrame or pandas.Series
            Training targets (self.output_size-day Bitcoin prices)
        cv : int, default=5
            Number of cross-validation folds
        return_train_score : bool, default=False
            Whether to return training scores
        verbose : bool, default=True
            Whether to show progress bar

        Returns:
        --------
        cv_results : dict
            Dictionary containing cross-validation results with RMSE scores
        """
        # Ensure y is properly formatted for self.output_size-day predictions
        if isinstance(y, pd.Series):
            # If single series, we need to reshape or create self.output_size-day targets
            # This assumes y contains sequential prices that need to be windowed
            y_array = y.values
            if len(y_array) < self.output_size:
                raise ValueError("Need at least self.output_size target values for self.output_size-day prediction")

            # Create self.output_size-day windows
            y_windowed = []
            for i in range(len(y_array) - 6):
                y_windowed.append(y_array[i:i + self.output_size])
            y = pd.DataFrame(y_windowed)

            # Adjust X to match the windowed y
            X = X.iloc[:len(y_windowed)]

        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != self.output_size:
                raise ValueError(f"y DataFrame must have self.output_size columns for self.output_size-day prediction, got {y.shape[1]}")
        else:
            y = np.array(y)
            if y.ndim == 1:
                raise ValueError("y must be 2D with self.output_size columns for self.output_size-day prediction")
            if y.shape[1] != self.output_size:
                raise ValueError(f"y must have self.output_size columns for self.output_size-day prediction, got {y.shape[1]}")
            y = pd.DataFrame(y)

        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        test_scores = []
        train_scores = [] if return_train_score else None
        fold_predictions = []
        fold_actuals = []

        splits = list(kfold.split(X))
        iterator = tqdm(enumerate(splits), total=cv, desc="CV Folds") if verbose else enumerate(splits)

        for fold, (train_idx, test_idx) in iterator:
            # Split data
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

            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_test = model.predict(X_test)

            # Ensure predictions are in correct format
            if isinstance(y_pred_test, list) and len(y_test) == 1:
                y_pred_test = np.array([y_pred_test])
            elif not isinstance(y_pred_test, np.ndarray):
                y_pred_test = np.array(y_pred_test)

            # Calculate RMSE using custom self.output_size-day RMSE function
            test_rmse = self._calculate_rmse_nday(y_test.values, y_pred_test)

            # Handle multiple samples in test set
            if isinstance(test_rmse, np.ndarray):
                test_score = np.mean(test_rmse)
            else:
                test_score = test_rmse

            test_scores.append(test_score)

            # Store predictions and actuals for detailed analysis
            fold_predictions.append(y_pred_test)
            fold_actuals.append(y_test.values)

            # Calculate training score if requested
            if return_train_score:
                y_pred_train = model.predict(X_train)
                if isinstance(y_pred_train, list) and len(y_train) == 1:
                    y_pred_train = np.array([y_pred_train])
                elif not isinstance(y_pred_train, np.ndarray):
                    y_pred_train = np.array(y_pred_train)

                train_rmse = self._calculate_rmse_nday(y_train.values, y_pred_train)
                if isinstance(train_rmse, np.ndarray):
                    train_score = np.mean(train_rmse)
                else:
                    train_score = train_rmse
                train_scores.append(train_score)

        # Compile results
        cv_results = {
            'test_scores': np.array(test_scores),
            'test_score_mean': np.mean(test_scores),
            'test_score_std': np.std(test_scores),
            'fold_predictions': fold_predictions,
            'fold_actuals': fold_actuals,
            'scoring_metric': 'self.output_size-day RMSE',
            'n_folds': cv
        }

        if return_train_score:
            cv_results.update({
                'train_scores': np.array(train_scores),
                'train_score_mean': np.mean(train_scores),
                'train_score_std': np.std(train_scores)
            })

        if verbose:
            print(f"\nCross-Validation Results (self.output_size-day Bitcoin Price RMSE):")
            print(f"Mean Test RMSE: {cv_results['test_score_mean']:.4f} (+/- {cv_results['test_score_std'] * 2:.4f})")
            if return_train_score:
                print(
                    f"Mean Train RMSE: {cv_results['train_score_mean']:.4f} (+/- {cv_results['train_score_std'] * 2:.4f})")
            print(f"Individual fold scores: {[f'{score:.4f}' for score in test_scores]}")

        return cv_results

    def cross_validate_detailed(self, X, y, cv=5, return_predictions=True):
        """
        Perform detailed cross-validation with per-day RMSE analysis

        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.DataFrame
            Training targets (self.output_size-day Bitcoin prices)
        cv : int, default=5
            Number of cross-validation folds
        return_predictions : bool, default=True
            Whether to return detailed predictions

        Returns:
        --------
        detailed_results : dict
            Detailed cross-validation results including per-day RMSE
        """
        cv_results = self.cross_validate(X, y, cv=cv, return_train_score=True, verbose=False)

        # Calculate per-day RMSE across all folds
        all_predictions = np.vstack(cv_results['fold_predictions'])
        all_actuals = np.vstack(cv_results['fold_actuals'])

        # Per-day RMSE
        per_day_rmse = []
        for day in range(self.output_size):
            day_rmse = np.sqrt(np.mean((all_actuals[:, day] - all_predictions[:, day]) ** 2))
            per_day_rmse.append(day_rmse)

        detailed_results = {
            **cv_results,
            'per_day_rmse': per_day_rmse,
            'day_labels': [f'Day_{i + 1}' for i in range(self.output_size)],
            'overall_rmse': cv_results['test_score_mean'],
            'best_day': np.argmin(per_day_rmse) + 1,
            'worst_day': np.argmax(per_day_rmse) + 1,
            'rmse_range': np.max(per_day_rmse) - np.min(per_day_rmse)
        }

        print(f"\nDetailed self.output_size-Day Bitcoin Price Prediction Analysis:")
        print(f"Overall RMSE: {detailed_results['overall_rmse']:.4f}")
        print(f"Best performing day: Day {detailed_results['best_day']} (RMSE: {np.min(per_day_rmse):.4f})")
        print(f"Worst performing day: Day {detailed_results['worst_day']} (RMSE: {np.max(per_day_rmse):.4f})")
        print(f"RMSE range across days: {detailed_results['rmse_range']:.4f}")
        print("\nPer-day RMSE:")
        for i, rmse in enumerate(per_day_rmse):
            print(f"  Day {i + 1}: {rmse:.4f}")

        return detailed_results

    def get_params(self, deep=True):
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

    def save(self, filepath):
        """Save the trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load a trained model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def model_info(self):
        """
        Get information about the current model state

        Returns:
        --------
        info : dict
            Dictionary containing model information
        """
        total_estimators = sum(len(day_est) for day_est in self.estimators_) if self.estimators_ else 0

        info = {
            'is_fitted': len(self.estimators_) > 0,
            'output_size': self.output_size,
            'n_days_predicted': self.output_size,
            'total_estimators_fitted': total_estimators,
            'n_estimators_per_day': [len(day_est) for day_est in self.estimators_] if self.estimators_ else [],
            'n_estimators_configured': self.n_estimators,
            'has_feature_importances': self.feature_importances_ is not None,
            'hyperparameters': self.get_params()
        }

        if self.train_scores_:
            info['final_training_rmse'] = self.train_scores_[-1]

        return info
