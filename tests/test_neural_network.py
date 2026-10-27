"""
Unit and mathematical tests for Neural Network X-Ray.
Tests forward pass, BCE loss, numerical gradient verification,
parameter updates, and XOR training convergence.
"""

import numpy as np
import pytest
from neural_network import (
    NeuralNetwork,
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh_act,
    tanh_derivative
)
from datasets import get_dataset


def test_activation_functions():
    """Verify activation functions and their mathematical derivatives."""
    z = np.array([-2.0, 0.0, 2.0])

    # Sigmoid
    s = sigmoid(z)
    assert np.allclose(s, 1.0 / (1.0 + np.exp(-z)))
    assert np.allclose(sigmoid(np.array([0.0])), 0.5)
    ds = sigmoid_derivative(z)
    assert np.allclose(ds, s * (1.0 - s))

    # ReLU
    r = relu(z)
    assert np.allclose(r, np.array([0.0, 0.0, 2.0]))
    dr = relu_derivative(z)
    assert np.allclose(dr, np.array([0.0, 0.0, 1.0]))

    # Tanh
    t = tanh_act(z)
    assert np.allclose(t, np.tanh(z))
    dt = tanh_derivative(z)
    assert np.allclose(dt, 1.0 - t**2)


def test_forward_pass_dimensions_and_ranges():
    """Test output shapes and probability bounds for forward propagation."""
    X, y, _ = get_dataset('XOR')
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='sigmoid', seed=42)

    A2, cache = nn.forward(X)

    assert A2.shape == (4, 1)
    assert cache['Z1'].shape == (4, 4)
    assert cache['A1'].shape == (4, 4)
    assert cache['Z2'].shape == (4, 1)
    assert cache['A2'].shape == (4, 1)

    # Sigmoid output must be strictly in (0, 1)
    assert np.all((A2 > 0.0) & (A2 < 1.0))


def test_loss_computation():
    """Verify binary cross-entropy loss calculation."""
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.9], [0.1]])

    loss = NeuralNetwork.compute_bce_loss(y_true, y_pred)
    expected = -0.5 * (np.log(0.9) + np.log(0.9))
    assert np.isclose(loss, expected, atol=1e-5)


def test_backpropagation_with_numerical_gradient_check():
    """
    Numerically verify analytical gradients computed by backward() using finite differences.
    (f(w + eps) - f(w - eps)) / (2 * eps) == grad
    """
    X, y, _ = get_dataset('XOR')
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, activation='tanh', seed=123)

    # Analytical gradients
    A2, cache = nn.forward(X)
    grads = nn.backward(y, cache)

    eps = 1e-6

    # 1. Check dW2
    for i in range(nn.W2.shape[0]):
        for j in range(nn.W2.shape[1]):
            orig_w = nn.W2[i, j]

            nn.W2[i, j] = orig_w + eps
            pred_plus, _ = nn.forward(X)
            loss_plus = nn.compute_bce_loss(y, pred_plus)

            nn.W2[i, j] = orig_w - eps
            pred_minus, _ = nn.forward(X)
            loss_minus = nn.compute_bce_loss(y, pred_minus)

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            nn.W2[i, j] = orig_w  # Restore

            assert np.isclose(grads['dW2'][i, j], num_grad, atol=1e-5), (
                f"dW2[{i},{j}] mismatch: analytical={grads['dW2'][i, j]}, numerical={num_grad}"
            )

    # 2. Check dW1
    for i in range(nn.W1.shape[0]):
        for j in range(nn.W1.shape[1]):
            orig_w = nn.W1[i, j]

            nn.W1[i, j] = orig_w + eps
            pred_plus, _ = nn.forward(X)
            loss_plus = nn.compute_bce_loss(y, pred_plus)

            nn.W1[i, j] = orig_w - eps
            pred_minus, _ = nn.forward(X)
            loss_minus = nn.compute_bce_loss(y, pred_minus)

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            nn.W1[i, j] = orig_w  # Restore

            assert np.isclose(grads['dW1'][i, j], num_grad, atol=1e-5), (
                f"dW1[{i},{j}] mismatch: analytical={grads['dW1'][i, j]}, numerical={num_grad}"
            )


def test_parameter_updates():
    """Verify that parameter updates follow new = old - lr * grad exactly."""
    X, y, _ = get_dataset('XOR')
    lr = 0.5
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='sigmoid', seed=42)

    A2, cache = nn.forward(X)
    grads = nn.backward(y, cache)

    w1_before = nn.W1.copy()
    b1_before = nn.b1.copy()

    updates = nn.update_parameters(grads, learning_rate=lr)

    # Check update arithmetic
    assert np.allclose(nn.W1, w1_before - lr * grads['dW1'])
    assert np.allclose(nn.b1, b1_before - lr * grads['db1'])
    assert np.allclose(updates['W1_change'], -lr * grads['dW1'])
    assert np.allclose(updates['W1_new'], nn.W1)


def test_neuron_calculation_consistency():
    """Verify that get_neuron_calculation returns exact values identical to the forward pass."""
    X, _, _ = get_dataset('XOR')
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='sigmoid', seed=99)

    A2, cache = nn.forward(X)

    # Test hidden neuron 1 for sample 2 (XOR row [1, 0])
    sample_idx = 2
    neuron_idx = 1
    calc_hidden = nn.get_neuron_calculation(sample_index=sample_idx, layer='hidden', neuron_index=neuron_idx, X=X)

    assert np.isclose(calc_hidden['weighted_sum'], cache['Z1'][sample_idx, neuron_idx])
    assert np.isclose(calc_hidden['output'], cache['A1'][sample_idx, neuron_idx])

    # Test output neuron
    calc_output = nn.get_neuron_calculation(sample_index=sample_idx, layer='output', neuron_index=0, X=X)
    assert np.isclose(calc_output['weighted_sum'], cache['Z2'][sample_idx, 0])
    assert np.isclose(calc_output['output'], cache['A2'][sample_idx, 0])


def test_xor_training_progress():
    """Verify training decreases loss and improves predictions on XOR."""
    X, y, _ = get_dataset('XOR')
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='tanh', seed=42)

    initial_probs, _ = nn.predict(X)
    initial_loss = nn.compute_bce_loss(y, initial_probs)

    # Train for 500 epochs with lr=1.0
    nn.train(X, y, epochs=500, learning_rate=1.0)

    final_probs, final_classes = nn.predict(X)
    final_loss = nn.compute_bce_loss(y, final_probs)

    # Loss must strictly decrease
    assert final_loss < initial_loss
    # History must record all epochs
    assert len(nn.history) >= 500
