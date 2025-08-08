"""
Part II: Modeling the Propagation Channel

This module implements atmospheric propagation modeling using ITU-R recommendations
via the ITU-Rpy library, including Free Space Path Loss, gaseous absorption,
rain fade, cloud attenuation, and tropospheric scintillation.
"""

import pandas as pd
import numpy as np
import itur
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AtmosphericChannel:
    """
    Models atmospheric propagation effects for satellite communications.
    
    This class calculates various atmospheric losses including Free Space Path Loss,
    gaseous absorption, rain fade, cloud attenuation, and scintillation using
    ITU-R standard models via the ITU-Rpy library.
    """
    
    def __init__(self):
        """Initialize the atmospheric channel model."""
        logger.info("Initializing AtmosphericChannel...")
        
        # ITU-Rpy uses astropy units, but we'll work primarily with raw values
        # and handle unit conversions internally
        
    def calculate_fspl(self, slant_range_km: np.ndarray, freq_GHz: float) -> np.ndarray:
        """
        Calculate Free Space Path Loss.
        
        Args:
            slant_range_km: Slant range distances in kilometers
            freq_GHz: Frequency in GHz
            
        Returns:
            FSPL in dB
        """
        # FSPL formula: 20*log10(d_km) + 20*log10(f_GHz) + 92.45
        fspl_dB = 20 * np.log10(slant_range_km) + 20 * np.log10(freq_GHz) + 92.45
        return fspl_dB
    
    def calculate_atmospheric_losses(self, geometry_df: pd.DataFrame, 
                                   ground_station_params: Dict[str, Any],
                                   link_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate all atmospheric loss components using ITU-R models.
        
        Args:
            geometry_df: DataFrame with elevation angles and other geometry data
            ground_station_params: Dict with 'lat', 'lon', 'alt_m' keys
            link_params: Dict with frequency and system parameters
            
        Returns:
            DataFrame with atmospheric loss components
        """
        logger.info("Calculating atmospheric losses using ITU-R models...")
        
        # Extract parameters
        lat = ground_station_params['lat']
        lon = ground_station_params['lon']
        alt_km = ground_station_params['alt_m'] / 1000.0  # Convert to km
        
        freq_GHz = link_params['freq_GHz']
        antenna_diameter_m = link_params.get('antenna_diameter_m', 1.2)
        rain_availability = link_params.get('rain_availability_percent', 99.99)
        
        # Convert availability to percentage of time exceeded
        rain_p = 100 - rain_availability
        
        # Extract elevation angles
        elevation_deg = geometry_df['elevation_deg'].values
        
        logger.info(f"Computing losses for {len(elevation_deg)} time steps")
        logger.info(f"Frequency: {freq_GHz} GHz, Location: {lat:.2f}°, {lon:.2f}°")
        
        try:
            # Use ITU-R atmospheric attenuation model
            # This function computes all major atmospheric loss components
            Ag, Ac, Ar, As, A_total = itur.atmospheric_attenuation_slant_path(
                lat=lat,
                lon=lon,
                f=freq_GHz,
                el=elevation_deg,
                p=rain_p,
                D=antenna_diameter_m,
                return_contributions=True
            )
            
            # Convert from astropy Quantity objects to numpy arrays
            gaseous_loss_dB = Ag.value if hasattr(Ag, 'value') else np.array(Ag)
            cloud_loss_dB = Ac.value if hasattr(Ac, 'value') else np.array(Ac)
            rain_fade_dB = Ar.value if hasattr(Ar, 'value') else np.array(Ar)
            scintillation_fade_dB = As.value if hasattr(As, 'value') else np.array(As)
            total_atmospheric_loss_dB = A_total.value if hasattr(A_total, 'value') else np.array(A_total)
            
        except Exception as e:
            logger.warning(f"ITU-R model failed: {e}. Using simplified models.")
            
            # Fallback to simplified models if ITU-Rpy fails
            gaseous_loss_dB = self._simple_gaseous_loss(elevation_deg, freq_GHz)
            cloud_loss_dB = self._simple_cloud_loss(elevation_deg, freq_GHz)
            rain_fade_dB = self._simple_rain_fade(elevation_deg, freq_GHz, rain_p)
            scintillation_fade_dB = self._simple_scintillation(elevation_deg, freq_GHz, antenna_diameter_m)
            total_atmospheric_loss_dB = gaseous_loss_dB + cloud_loss_dB + rain_fade_dB + scintillation_fade_dB
        
        # Create DataFrame with loss components
        loss_df = pd.DataFrame({
            'gaseous_loss_dB': gaseous_loss_dB,
            'cloud_loss_dB': cloud_loss_dB, 
            'rain_fade_dB': rain_fade_dB,
            'scintillation_fade_dB': scintillation_fade_dB,
            'total_atmospheric_loss_dB': total_atmospheric_loss_dB
        }, index=geometry_df.index)
        
        logger.info("Atmospheric loss calculation completed")
        return loss_df
    
    def _simple_gaseous_loss(self, elevation_deg: np.ndarray, freq_GHz: float) -> np.ndarray:
        """Simplified gaseous absorption model."""
        # Very basic model - zenith attenuation scaled by air mass
        zenith_loss_dB = 0.1 * (freq_GHz / 10.0) ** 1.5  # Rough frequency scaling
        air_mass = 1.0 / np.sin(np.radians(np.maximum(elevation_deg, 1.0)))
        return zenith_loss_dB * air_mass
    
    def _simple_cloud_loss(self, elevation_deg: np.ndarray, freq_GHz: float) -> np.ndarray:
        """Simplified cloud attenuation model."""
        # Basic cloud loss - increases with frequency and air mass
        zenith_cloud_loss = 0.05 * (freq_GHz / 10.0) ** 2
        air_mass = 1.0 / np.sin(np.radians(np.maximum(elevation_deg, 1.0)))
        return zenith_cloud_loss * air_mass
    
    def _simple_rain_fade(self, elevation_deg: np.ndarray, freq_GHz: float, 
                         rain_p: float) -> np.ndarray:
        """Simplified rain fade model."""
        # Rain fade increases dramatically with frequency and at low elevations
        # This is a very simplified model
        rain_rate_mm_hr = 10.0 * (rain_p / 0.01) ** 0.5  # Rough rain rate estimate
        specific_attenuation = 0.001 * (freq_GHz ** 2.2) * (rain_rate_mm_hr ** 1.2)
        
        # Path length through rain cell
        rain_height_km = 3.0  # Typical rain cell height
        path_length_km = rain_height_km / np.sin(np.radians(np.maximum(elevation_deg, 1.0)))
        
        return specific_attenuation * path_length_km
    
    def _simple_scintillation(self, elevation_deg: np.ndarray, freq_GHz: float,
                            antenna_diameter_m: float) -> np.ndarray:
        """Simplified scintillation model."""
        # Scintillation is worst at low elevations and high frequencies
        # and decreases with larger antennas
        scint_variance = (freq_GHz / 10.0) ** 2.2 / (antenna_diameter_m ** 1.2)
        elevation_factor = 1.0 / np.sin(np.radians(np.maximum(elevation_deg, 1.0))) ** 1.2
        
        # Return RMS scintillation (could be scaled for different percentiles)
        return 2.0 * np.sqrt(scint_variance * elevation_factor)
    
    def calculate_total_path_loss(self, geometry_df: pd.DataFrame,
                                ground_station_params: Dict[str, Any],
                                link_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate total path loss combining FSPL and atmospheric effects.
        
        Args:
            geometry_df: DataFrame with geometry data including slant_range_km
            ground_station_params: Ground station parameters
            link_params: Link parameters including frequency
            
        Returns:
            DataFrame with all loss components and total path loss
        """
        logger.info("Calculating total path loss...")
        
        # Calculate Free Space Path Loss
        fspl_dB = self.calculate_fspl(
            geometry_df['slant_range_km'].values,
            link_params['freq_GHz']
        )
        
        # Calculate atmospheric losses
        atm_losses_df = self.calculate_atmospheric_losses(
            geometry_df, ground_station_params, link_params
        )
        
        # Combine all losses
        total_loss_df = atm_losses_df.copy()
        total_loss_df['fspl_dB'] = fspl_dB
        total_loss_df['total_path_loss_dB'] = (
            fspl_dB + atm_losses_df['total_atmospheric_loss_dB']
        )
        
        logger.info("Total path loss calculation completed")
        return total_loss_df
    
    def analyze_loss_statistics(self, loss_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze loss statistics for the pass.
        
        Args:
            loss_df: DataFrame with loss components
            
        Returns:
            Dictionary with loss statistics
        """
        stats = {}
        
        for column in loss_df.columns:
            if column.endswith('_dB'):
                stats[column] = {
                    'min': float(loss_df[column].min()),
                    'max': float(loss_df[column].max()),
                    'mean': float(loss_df[column].mean()),
                    'std': float(loss_df[column].std()),
                    'median': float(loss_df[column].median())
                }
        
        logger.info("Loss statistics analysis completed")
        return stats


def create_default_link_params() -> Dict[str, Any]:
    """
    Create default link parameters for Ku-band satellite communication.
    
    Returns:
        Dictionary with default link parameters
    """
    return {
        'freq_GHz': 20.0,                    # Ku-band downlink
        'antenna_diameter_m': 1.2,           # VSAT antenna
        'rain_availability_percent': 99.99   # 99.99% availability target
    }


def create_default_ground_station_params() -> Dict[str, Any]:
    """
    Create default ground station parameters for Delhi, India.
    
    Returns:
        Dictionary with default ground station parameters
    """
    return {
        'lat': 28.61,      # Delhi latitude
        'lon': 77.23,      # Delhi longitude  
        'alt_m': 216.0     # Delhi altitude in meters
    } 