"""
LEO Satellite Communication Link Simulation

A comprehensive simulation framework for modeling Low Earth Orbit (LEO) satellite 
communication links, including orbital mechanics, atmospheric propagation, and 
link budget calculations for AI-driven adaptive coding and modulation research.
"""

__version__ = "1.0.0"
__author__ = "LEO Simulation Framework"

from .geometry import SatelliteGeometry
from .propagation import AtmosphericChannel
from .link_budget import LinkBudget
from .data_generator import SimulationDataGenerator

__all__ = [
    'SatelliteGeometry',
    'AtmosphericChannel', 
    'LinkBudget',
    'SimulationDataGenerator'
] 