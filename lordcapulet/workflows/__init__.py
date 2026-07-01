"""
AiiDA workflow plugins for LordCapulet.
"""

from .standard_magnetic_scan import StandardMagneticScanWorkChain
from .constrained_scan import ConstrainedScanWorkChain
from .global_constrained_search import GlobalConstrainedSearchWorkChain

# Backwards-compatibility alias
AFMScanWorkChain = StandardMagneticScanWorkChain

__all__ = [
    'StandardMagneticScanWorkChain',
    'AFMScanWorkChain',  # deprecated alias
    'ConstrainedScanWorkChain',
    'GlobalConstrainedSearchWorkChain',
]
