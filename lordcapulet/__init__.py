"""
LordCapulet: AiiDA plugins for constrained DFT+U calculations.

This package provides AiiDA plugins for running constrained DFT+U calculations
with Quantum ESPRESSO, including workflows for scanning different magnetic
configurations and target occupation matrices.
"""

__version__ = "0.1.0"
__author__ = "Alberto Carta"
__email__ = "your.email@example.com"

# Import main classes for convenient access
from .calculations.constrained_pw import ConstrainedPWCalculation
from .workflows.standard_magnetic_scan import StandardMagneticScanWorkChain
from .workflows.constrained_scan import ConstrainedScanWorkChain
from .workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

# Backwards-compatibility alias
AFMScanWorkChain = StandardMagneticScanWorkChain

__all__ = [
    'ConstrainedPWCalculation',
    'StandardMagneticScanWorkChain',
    'AFMScanWorkChain',  # deprecated alias
    'ConstrainedScanWorkChain',
    'GlobalConstrainedSearchWorkChain',
]
