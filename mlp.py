"""
MLP (Multi-Layer Perceptron).
Interface compatible with grid_search (fit/predict).
"""

import math
import numpy as np


# =============================================================================
# Activation functions (implemented manually with numpy)
# =============================================================================

def _relu(x):
    """ReLU activation: f(x) = max(0, x)"""
    return np.maximum(0.0, x)


def _relu_derivative(x):
    """Derivative of ReLU: f'(x) = 1 if x > 0 else 0"""
    return (x > 0).astype(np.float64)


def _tanh(x):
    """Hyperbolic tangent: f(x) = (e^x - e^-x) / (e^x + e^-x)"""
    return np.tanh(x)


def _tanh_derivative(x):
    """Derivative of tanh: f'(x) = 1 - f(x)^2"""
    fx = np.tanh(x)
    return 1.0 - fx * fx


def _sigmoid(x):
    """Sigmoid activation: f(x) = 1 / (1 + e^-x)"""
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_derivative(x):
    """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
    fx = _sigmoid(x)
    return fx * (1.0 - fx)


# Map activation names to functions
_ACTIVATIONS = {
    'relu': (_relu, _relu_derivative),
    'tanh': (_tanh, _tanh_derivative),
}


# =============================================================================
# MLP Classifier
# =============================================================================

class MLPClassifier:
    """
    Multi-Layer Perceptron Classifier with manual backpropagation.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        Sizes of hidden layers, e.g. (50,) or (100, 50).
    activation : str, default='relu'
        Activation function for hidden layers: 'relu' or 'tanh'.
    alpha : float, default=0.0001
        L2 regularization strength.
    max_iter : int, default=200
        Maximum number of epochs.
    learning_rate_init : float, default=0.01
        Initial learning rate.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 alpha=0.0001, max_iter=200, learning_rate_init=0.01, **params):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        
        # Store extra params (for grid_search compatibility)
        for key, value in params.items():
            setattr(self, key, value)
        
        # Internal state (initialized in fit)
        self.weights_ = None
        self.biases_ = None
        self.loss_history_ = None
        self._mean = None
        self._std = None
    
    def _initialize_weights(self, n_features):
        """He initialization: normal distribution scaled by sqrt(2 / fan_in)."""
        weights = []
        biases = []
        
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # He initialization
            std = math.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))
            
            weights.append(W)
            biases.append(b)
        
        return weights, biases
    
    def _normalize(self, X, fit=True):
        """Z-score normalization."""
        X = np.asarray(X, dtype=np.float64)
        if fit:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            # Avoid division by zero
            self._std[self._std == 0] = 1.0
        
        return (X - self._mean) / self._std
    
    def _forward(self, X):
        """Forward pass through the network.
        
        Returns (activations, linear_outputs) where:
        - activations: list of post-activation values at each layer
        - linear_outputs: list of pre-activation values (Z) at each layer
        """
        act_fn, _ = _ACTIVATIONS[self.activation]
        
        activations = [X]      # A0 = input
        linear_outputs = []    # Z1, Z2, ...
        
        n_layers = len(self.weights_)
        
        for i in range(n_layers - 1):
            # Hidden layer
            Z = activations[-1] @ self.weights_[i] + self.biases_[i]
            linear_outputs.append(Z)
            A = act_fn(Z)
            activations.append(A)
        
        # Output layer (sigmoid)
        Z_out = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        linear_outputs.append(Z_out)
        A_out = _sigmoid(Z_out)
        activations.append(A_out)
        
        return activations, linear_outputs
    
    def _backward(self, activations, linear_outputs, y):
        """Backward pass to compute gradients.
        
        Returns lists of weight gradients and bias gradients.
        """
        act_fn, act_deriv = _ACTIVATIONS[self.activation]
        n_samples = y.shape[0]
        n_layers = len(self.weights_)
        
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        
        # Output layer error (binary cross-entropy + sigmoid => ŷ - y)
        A_out = activations[-1]
        delta = A_out - y.reshape(-1, 1)  # shape: (n_samples, 1)
        
        # Gradient for output layer
        weight_grads[-1] = (activations[-2].T @ delta) / n_samples
        bias_grads[-1] = np.sum(delta, axis=0, keepdims=True) / n_samples
        
        # Backpropagate through hidden layers
        for i in range(n_layers - 2, -1, -1):
            # delta = delta @ W^T * f'(Z)
            delta = delta @ self.weights_[i + 1].T * act_deriv(linear_outputs[i])
            
            weight_grads[i] = (activations[i].T @ delta) / n_samples
            bias_grads[i] = np.sum(delta, axis=0, keepdims=True) / n_samples
        
        return weight_grads, bias_grads
    
    def _update_weights(self, weight_grads, bias_grads, lr):
        """Update weights and biases with gradients and L2 regularization."""
        for i in range(len(self.weights_)):
            # L2 regularization: subtract alpha * W
            self.weights_[i] -= lr * (weight_grads[i] + self.alpha * self.weights_[i])
            self.biases_[i] -= lr * bias_grads[i]
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss: -1/N * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]"""
        n = len(y_true)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        
        # Clip to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """
        Train the MLP using backpropagation.
        
        Parameters
        ----------
        X : list of lists or array-like, shape (n_samples, n_features)
            Training data.
        y : list or array-like, shape (n_samples,)
            Target values (0 or 1).
            
        Returns
        -------
        self : object
            Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Normalize features
        X = self._normalize(X, fit=True)
        
        # Initialize weights
        self.weights_, self.biases_ = self._initialize_weights(X.shape[1])
        
        self.loss_history_ = []
        lr = self.learning_rate_init
        
        for epoch in range(self.max_iter):
            # Forward pass
            activations, linear_outputs = self._forward(X)
            
            # Compute loss
            y_pred = activations[-1]
            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history_.append(loss)
            
            # Backward pass
            weight_grads, bias_grads = self._backward(activations, linear_outputs, y)
            
            # Update weights
            self._update_weights(weight_grads, bias_grads, lr)
            
            # Check for NaN/Inf
            if np.isnan(loss) or np.isinf(loss):
                break
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for input samples.
        
        Parameters
        ----------
        X : list of lists or array-like, shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : list of int
            Predicted class labels (0 or 1). Threshold = 0.5.
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Normalize using training statistics
        X = self._normalize(X, fit=False)
        
        # Forward pass
        activations, _ = self._forward(X)
        
        # Threshold at 0.5
        probabilities = activations[-1].flatten()
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions.tolist()
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : list of lists or array-like, shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : list of float
            Probability of class 1 for each sample.
        """
        X = np.asarray(X, dtype=np.float64)
        X = self._normalize(X, fit=False)
        activations, _ = self._forward(X)
        return activations[-1].flatten().tolist()


# =============================================================================
# Test example
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MLP Classifier from Scratch - Test")
    print("=" * 60)
    
    # Create a simple XOR-like dataset (not linearly separable)
    np.random.seed(42)
    n_samples = 200
    
    # Generate two clusters for class 0
    X0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-2, -2])
    X1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, 2])
    
    # Generate two clusters for class 1
    X2 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-2, 2])
    X3 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, -2])
    
    X = np.vstack([X0, X1, X2, X3])
    y = np.array([0] * n_samples + [1] * n_samples)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Train MLP
    print("Training MLP with hidden_layer_sizes=(8, 4), activation='relu'...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation='relu',
        alpha=0.0001,
        max_iter=500,
        learning_rate_init=0.1
    )
    mlp.fit(X_train.tolist(), y_train.tolist())
    
    # Predict
    y_pred = mlp.predict(X_test.tolist())
    y_test_list = y_test.tolist()
    
    # Accuracy
    correct = sum(1 for p, t in zip(y_pred, y_test_list) if p == t)
    accuracy = correct / len(y_test) * 100
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(y_test)}")
    
    # Check predict_proba
    proba = mlp.predict_proba(X_test[:5].tolist())
    print(f"\nSample probabilities (first 5): {[f'{p:.4f}' for p in proba]}")
    print(f"Sample predictions (first 5): {y_pred[:5]}")
    print(f"Sample true labels (first 5): {y_test_list[:5]}")
    
    # Loss curve summary
    print(f"\nInitial loss: {mlp.loss_history_[0]:.4f}")
    print(f"Final loss: {mlp.loss_history_[-1]:.4f}")
    
    # Test with tanh
    print("\n" + "=" * 60)
    print("Testing with tanh activation...")
    mlp_tanh = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation='tanh',
        alpha=0.0001,
        max_iter=500,
        learning_rate_init=0.1
    )
    mlp_tanh.fit(X_train.tolist(), y_train.tolist())
    y_pred_tanh = mlp_tanh.predict(X_test.tolist())
    correct_tanh = sum(1 for p, t in zip(y_pred_tanh, y_test_list) if p == t)
    accuracy_tanh = correct_tanh / len(y_test) * 100
    print(f"Test Accuracy (tanh): {accuracy_tanh:.2f}%")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
