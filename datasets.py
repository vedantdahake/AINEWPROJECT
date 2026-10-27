"""
Dataset definitions for Neural Network X-Ray.
Provides canonical datasets like XOR, AND, OR, and Circle for classification experiments.
"""

from typing import Tuple, Dict, Any
import numpy as np


DATASETS: Dict[str, Dict[str, Any]] = {
    'XOR': {
        'name': 'XOR (Exclusive OR)',
        'description': (
            'The classic non-linearly separable problem. Output is 1 if and only if '
            'exactly one input is 1. Requires at least one hidden layer to solve.'
        ),
        'X': np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=float),
        'y': np.array([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ], dtype=float),
        'feature_names': ['x₁ (Input 1)', 'x₂ (Input 2)'],
        'target_name': 'y (Expected)',
    },
    'AND': {
        'name': 'AND Logic Gate',
        'description': (
            'Linearly separable logic gate. Output is 1 only when both inputs are 1.'
        ),
        'X': np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=float),
        'y': np.array([
            [0.0],
            [0.0],
            [0.0],
            [1.0]
        ], dtype=float),
        'feature_names': ['x₁ (Input 1)', 'x₂ (Input 2)'],
        'target_name': 'y (Expected)',
    },
    'OR': {
        'name': 'OR Logic Gate',
        'description': (
            'Linearly separable logic gate. Output is 1 if at least one input is 1.'
        ),
        'X': np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=float),
        'y': np.array([
            [0.0],
            [1.0],
            [1.0],
            [1.0]
        ], dtype=float),
        'feature_names': ['x₁ (Input 1)', 'x₂ (Input 2)'],
        'target_name': 'y (Expected)',
    },
    'Circle': {
        'name': 'Concentric Circle',
        'description': (
            'Non-linear classification benchmark. Points inside radius 0.5 are class 1, outside are 0.'
        ),
        'X': np.array([
            [0.1, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.9, 0.9],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.9, 0.1]
        ], dtype=float),
        'y': np.array([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0]
        ], dtype=float),
        'feature_names': ['x₁ (Coordinate 1)', 'x₂ (Coordinate 2)'],
        'target_name': 'y (Expected)',
    }
}


def get_dataset(name: str = 'XOR') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Retrieve dataset features X, target y, and metadata dictionary.
    """
    key = name.upper() if name.upper() in DATASETS else 'XOR'
    info = DATASETS[key]
    return info['X'].copy(), info['y'].copy(), info
