"""
Neural Network Module
=====================

Conditional neural networks for parameter prediction.

- ConditionalNetwork: Predicts (A, B) from packing fraction η
- FeatureExtractor: Extracts local structural features from density
"""

from .network import ConditionalNetwork
from .features import FeatureExtractor

__all__ = ['ConditionalNetwork', 'FeatureExtractor']
