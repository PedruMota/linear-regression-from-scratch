import numpy as np
import pandas as pd

class LinearRegressionGD:
    """
    A custom Linear Regression implementation using Gradient Descent.
    Supports automatic feature scaling and one-hot encoding for categorical data.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, normalize=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.normalize = normalize
        
        # Model Parameters
        self.weights = None
        self.bias = None
        self.cost_history = []
        
        # Engineering Metadata (to ensure train/test consistency)
        self.training_columns = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        # One-Hot Encoding (Handling Categorical Data)
        X_processed = pd.get_dummies(X, drop_first=True)
        X_processed = X_processed.astype(float)
        
        # Save columns to enforce schema consistency during prediction
        self.training_columns = X_processed.columns
        X_matrix = X_processed.values

        # Feature Scaling (Z-Score Normalization)
        if self.normalize:
            # Calculate and store mean/std from the training set
            self.mean = np.mean(X_matrix, axis=0)
            self.std = np.std(X_matrix, axis=0)
            
            # Add epsilon to prevent division by zero in constant columns
            self.std[self.std == 0] = 1e-8
            
            # Apply scaling: z = (x - mean) / std
            X_matrix = (X_matrix - self.mean) / self.std

        # Parameter Initialization
        n_samples, n_features = X_matrix.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        # Gradient Descent Loop
        for _ in range(self.n_iterations):
            # Prediction
            y_predicted = np.dot(X_matrix, self.weights) + self.bias
            residuals = y_predicted - y
            
            # Gradient Calculation
            dw = (1 / n_samples) * np.dot(X_matrix.T, residuals)
            db = (1 / n_samples) * np.sum(residuals)
            
            # Update Weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track Cost (MSE)
            cost = np.mean(residuals ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        # Processing & Alignment
        X_processed = pd.get_dummies(X, drop_first=True)

        # Enforce the same columns as training (fill missing with 0)
        X_processed = X_processed.reindex(columns=self.training_columns, fill_value=0)
        X_processed = X_processed.astype(float)
        X_matrix = X_processed.values
        
        # Feature Scaling (using stored training stats)
        if self.normalize:
            X_matrix = (X_matrix - self.mean) / self.std
        
        return np.dot(X_matrix, self.weights) + self.bias