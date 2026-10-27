"""
Neural Network implementation using NumPy for educational visualization.
Calculates exact forward propagation, loss, backpropagation, and parameter updates.
No external deep learning frameworks (TensorFlow, PyTorch) are used.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute numerically stable sigmoid activation."""
    # Clip z to prevent overflow/underflow in exp
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """Compute sigmoid derivative: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    if a is None:
        a = sigmoid(z)
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    """Compute ReLU activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """Compute ReLU derivative: 1 if z > 0 else 0."""
    return (z > 0).astype(float)


def tanh_act(z: np.ndarray) -> np.ndarray:
    """Compute Tanh activation: tanh(z)."""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
    """Compute Tanh derivative: 1 - tanh^2(z)."""
    if a is None:
        a = np.tanh(z)
    return 1.0 - np.power(a, 2)


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh_act, tanh_derivative),
}


class NeuralNetwork:
    """
    A 2-layer fully connected feedforward neural network:
    Input (2) -> Hidden (H) -> Output (1)

    Designed specifically for educational inspection:
    - Pure NumPy calculations
    - Transparent forward pass and gradient calculation
    - Detailed neuron inspection and parameter update tracking
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 4,
        output_size: int = 1,
        activation: str = 'sigmoid',
        seed: int = 42
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.seed = seed

        if self.activation_name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from: {list(ACTIVATIONS.keys())}")

        self.act_fn, self.act_deriv = ACTIVATIONS[self.activation_name]
        self.history: List[Dict[str, Any]] = []
        self.last_cache: Dict[str, np.ndarray] = {}
        self.last_grads: Dict[str, np.ndarray] = {}
        self.last_updates: Dict[str, Any] = {}

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases reproducibly using the random seed."""
        rng = np.random.RandomState(self.seed)

        # Xavier/Glorot initialization for smooth educational demonstration
        limit_w1 = np.sqrt(6.0 / (self.input_size + self.hidden_size))
        self.W1 = rng.uniform(-limit_w1, limit_w1, size=(self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))

        limit_w2 = np.sqrt(6.0 / (self.hidden_size + self.output_size))
        self.W2 = rng.uniform(-limit_w2, limit_w2, size=(self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))

        self.history = []
        self.last_cache = {}
        self.last_grads = {}
        self.last_updates = {}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform forward propagation.
        X: shape (N, input_size)

        Returns:
            A2: predictions (N, output_size)
            cache: dictionary containing Z1, A1, Z2, A2, and X
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Layer 1 (Hidden layer)
        # Z1 = X * W1 + b1
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.act_fn(Z1)

        # Layer 2 (Output layer with Sigmoid)
        # Z2 = A1 * W2 + b2
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)

        cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        self.last_cache = cache
        return A2, cache

    @staticmethod
    def compute_bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss:
        L = - (1/N) * sum( y * log(p) + (1-y) * log(1-p) )
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        return float(loss)

    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute classification accuracy based on decision threshold.
        Returns accuracy as percentage (0.0 to 100.0).
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
        pred_class = (y_pred >= threshold).astype(int)
        correct = np.sum(pred_class == y_true.astype(int))
        return float((correct / len(y_true)) * 100.0)

    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute gradients via backpropagation.
        y_true: shape (N, 1)
        cache: cache from forward pass

        Returns:
            grads: dictionary of dW1, db1, dW2, db2
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        X = cache['X']
        Z1 = cache['Z1']
        A1 = cache['A1']
        A2 = cache['A2']
        N = X.shape[0]

        # Output error: dL/dZ2 for Sigmoid + BCE = A2 - y
        dZ2 = A2 - y_true  # (N, 1)
        dW2 = np.dot(A1.T, dZ2) / N  # (hidden_size, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N  # (1, 1)

        # Hidden layer error: dL/dZ1 = (dZ2 * W2^T) * act_deriv(Z1)
        dA1 = np.dot(dZ2, self.W2.T)  # (N, hidden_size)
        dZ1 = dA1 * self.act_deriv(Z1, A1)  # (N, hidden_size)
        dW1 = np.dot(X.T, dZ1) / N  # (input_size, hidden_size)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N  # (1, hidden_size)

        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2,
            'dZ1': dZ1,
            'dZ2': dZ2
        }
        self.last_grads = grads
        return grads

    def update_parameters(self, grads: Dict[str, np.ndarray], learning_rate: float) -> Dict[str, Any]:
        """
        Update model parameters using gradient descent:
        new_param = old_param - learning_rate * grad

        Records parameter changes for educational display.
        """
        updates = {
            'learning_rate': learning_rate,
            'W1_old': self.W1.copy(),
            'b1_old': self.b1.copy(),
            'W2_old': self.W2.copy(),
            'b2_old': self.b2.copy(),
            'dW1': grads['dW1'].copy(),
            'db1': grads['db1'].copy(),
            'dW2': grads['dW2'].copy(),
            'db2': grads['db2'].copy(),
            'W1_change': -learning_rate * grads['dW1'],
            'b1_change': -learning_rate * grads['db1'],
            'W2_change': -learning_rate * grads['dW2'],
            'b2_change': -learning_rate * grads['db2'],
        }

        self.W1 += updates['W1_change']
        self.b1 += updates['b1_change']
        self.W2 += updates['W2_change']
        self.b2 += updates['b2_change']

        updates['W1_new'] = self.W1.copy()
        updates['b1_new'] = self.b1.copy()
        updates['W2_new'] = self.W2.copy()
        updates['b2_new'] = self.b2.copy()

        self.last_updates = updates
        return updates

    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> Dict[str, Any]:
        """
        Execute a single gradient descent step on the dataset.
        Returns loss, accuracy, and step details.
        """
        A2, cache = self.forward(X)
        loss = self.compute_bce_loss(y, A2)
        acc = self.compute_accuracy(y, A2)

        grads = self.backward(y, cache)
        updates = self.update_parameters(grads, learning_rate)

        # Compute new forward pass to record post-update metrics
        new_A2, _ = self.forward(X)
        new_loss = self.compute_bce_loss(y, new_A2)
        new_acc = self.compute_accuracy(y, new_A2)

        epoch_num = len(self.history) + 1
        record = {
            'epoch': epoch_num,
            'loss': new_loss,
            'accuracy': new_acc,
            'loss_before': loss,
            'accuracy_before': acc,
        }
        self.history.append(record)
        return {
            'record': record,
            'updates': updates,
            'grads': grads
        }

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> List[Dict[str, Any]]:
        """
        Train the network for a specified number of epochs.
        Returns training history.
        """
        # If this is starting from fresh epoch 0, record initial state
        if len(self.history) == 0:
            init_pred, _ = self.forward(X)
            init_loss = self.compute_bce_loss(y, init_pred)
            init_acc = self.compute_accuracy(y, init_pred)
            self.history.append({
                'epoch': 0,
                'loss': init_loss,
                'accuracy': init_acc,
                'loss_before': init_loss,
                'accuracy_before': init_acc,
            })

        for _ in range(epochs):
            A2, cache = self.forward(X)
            grads = self.backward(y, cache)
            self.update_parameters(grads, learning_rate)

            # Evaluate after update
            post_pred, _ = self.forward(X)
            loss = self.compute_bce_loss(y, post_pred)
            acc = self.compute_accuracy(y, post_pred)

            self.history.append({
                'epoch': len(self.history),
                'loss': loss,
                'accuracy': acc,
            })

        return self.history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predicted probabilities and predicted binary classes.
        """
        probs, _ = self.forward(X)
        classes = (probs >= threshold).astype(int)
        return probs, classes

    def get_neuron_calculation(
        self,
        sample_index: int,
        layer: str,
        neuron_index: int,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract detailed arithmetic breakdown for a specific neuron and sample.
        layer: 'hidden' or 'output'
        neuron_index: 0-indexed integer
        sample_index: 0-indexed integer row in X

        Returns:
            Dictionary with inputs, weights, products, bias, z, act_fn_name, and final output a,
            along with a formatted educational formula string.
        """
        X = np.asarray(X, dtype=float)
        # Ensure fresh forward pass for this data
        A2, cache = self.forward(X)

        if layer == 'hidden':
            if neuron_index < 0 or neuron_index >= self.hidden_size:
                raise IndexError(f"Hidden neuron index {neuron_index} out of bounds (0..{self.hidden_size-1})")

            inputs = cache['X'][sample_index].tolist()
            input_labels = [f"x{i+1}" for i in range(len(inputs))]
            weights = self.W1[:, neuron_index].tolist()
            weight_labels = [f"w{i+1},{neuron_index+1}" for i in range(len(weights))]
            products = [i * w for i, w in zip(inputs, weights)]
            bias = float(self.b1[0, neuron_index])
            weighted_sum = float(sum(products) + bias)

            act_name = self.activation_name
            output_val = float(cache['A1'][sample_index, neuron_index])

            # Formula formatted string
            terms = [f"({inputs[i]:.2f} × {weights[i]:+.3f})" for i in range(len(inputs))]
            sum_str = " + ".join(terms) + f" + ({bias:+.3f})"
            calc_formula = f"z = {sum_str} = {weighted_sum:.4f}\na = {act_name}({weighted_sum:.4f}) = {output_val:.4f}"

            explanation = (
                f"Neuron H{neuron_index+1} takes the {len(inputs)} external input values, "
                f"multiplies each by its connection weight, adds bias ({bias:+.3f}), "
                f"and applies the {act_name} activation function."
            )

            return {
                'layer_name': f"Hidden Layer (Neuron H{neuron_index+1})",
                'input_labels': input_labels,
                'inputs': inputs,
                'weight_labels': weight_labels,
                'weights': weights,
                'products': products,
                'bias': bias,
                'weighted_sum': weighted_sum,
                'activation_name': act_name,
                'output': output_val,
                'formula_str': calc_formula,
                'explanation': explanation,
            }

        elif layer == 'output':
            inputs = cache['A1'][sample_index].tolist()
            input_labels = [f"h{i+1}" for i in range(len(inputs))]
            weights = self.W2[:, neuron_index].tolist()
            weight_labels = [f"w_h{i+1},out" for i in range(len(weights))]
            products = [i * w for i, w in zip(inputs, weights)]
            bias = float(self.b2[0, neuron_index])
            weighted_sum = float(sum(products) + bias)

            act_name = "sigmoid"
            output_val = float(cache['A2'][sample_index, neuron_index])

            terms = [f"({inputs[i]:.3f} × {weights[i]:+.3f})" for i in range(len(inputs))]
            sum_str = " + ".join(terms) + f" + ({bias:+.3f})"
            calc_formula = f"z = {sum_str} = {weighted_sum:.4f}\na = sigmoid({weighted_sum:.4f}) = {output_val:.4f}"

            explanation = (
                f"The Output Neuron takes activations from all {len(inputs)} hidden neurons as inputs, "
                f"multiplies each by its weight, adds the bias ({bias:+.3f}), "
                f"and applies sigmoid to calculate the predicted probability (0 to 1)."
            )

            return {
                'layer_name': "Output Layer (Neuron ŷ)",
                'input_labels': input_labels,
                'inputs': inputs,
                'weight_labels': weight_labels,
                'weights': weights,
                'products': products,
                'bias': bias,
                'weighted_sum': weighted_sum,
                'activation_name': act_name,
                'output': output_val,
                'formula_str': calc_formula,
                'explanation': explanation,
            }
        else:
            raise ValueError(f"Unknown layer '{layer}'. Must be 'hidden' or 'output'.")
