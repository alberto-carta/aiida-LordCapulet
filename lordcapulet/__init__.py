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
from .workflows.afm_scan import AFMScanWorkChain
from .workflows.constrained_scan import ConstrainedScanWorkChain

__all__ = [
    'ConstrainedPWCalculation',
    'AFMScanWorkChain', 
    'ConstrainedScanWorkChain',
]
