"""
Surrogate quality prediction for QD search.

Enables quality prediction BEFORE audio rendering using genome features.
"""

from .surrogate_network import SurrogateEnsemble, SurrogateMember, create_surrogate

__all__ = [
    'SurrogateEnsemble',
    'SurrogateMember',
    'create_surrogate'
]
