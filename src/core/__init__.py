"""
Core satellite communication modules.

This module contains the fundamental components for satellite communication
including signal processing, link optimization, and protocol management.
"""

from .signal_processor import SignalProcessor
from .link_optimizer import LinkOptimizer
from .protocol_manager import ProtocolManager
from .error_correction import ErrorCorrection

__all__ = [
    "SignalProcessor",
    "LinkOptimizer", 
    "ProtocolManager",
    "ErrorCorrection"
] 