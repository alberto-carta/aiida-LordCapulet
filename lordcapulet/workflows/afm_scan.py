"""Compatibility shim - AFMScanWorkChain has been renamed to StandardMagneticScanWorkChain.

This module is kept so that existing AiiDA database nodes (which store
``lordcapulet.afm_scan`` as their process_type) can still be recognised.
New code should import ``StandardMagneticScanWorkChain`` directly.
"""
from lordcapulet.workflows.standard_magnetic_scan import StandardMagneticScanWorkChain

# Backwards-compatibility alias
AFMScanWorkChain = StandardMagneticScanWorkChain

__all__ = ['AFMScanWorkChain', 'StandardMagneticScanWorkChain']
