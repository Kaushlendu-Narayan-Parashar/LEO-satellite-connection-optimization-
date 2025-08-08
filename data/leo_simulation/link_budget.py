"""
Part III: The Satellite Link Budget as a Computational Graph

This module implements a comprehensive link budget calculation system using
a Directed Acyclic Graph (DAG) approach for satellite communication links.
It calculates EIRP, G/T, and C/N0 with proper dependency management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LinkBudgetNode(Enum):
    """Enumeration of all link budget calculation nodes."""
    # Input geometry and channel parameters
    ELEVATION_DEG = "elevation_deg"
    SLANT_RANGE_KM = "slant_range_km" 
    TOTAL_PATH_LOSS_DB = "total_path_loss_dB"
    RAIN_FADE_DB = "rain_fade_dB"
    
    # Static link parameters
    FREQ_GHZ = "freq_GHz"
    SAT_EIRP_DBW = "sat_eirp_dBW"
    GS_ANT_DIAM_M = "gs_ant_diam_m"
    GS_ANT_EFF = "gs_ant_eff"
    GS_LNB_NOISE_TEMP_K = "gs_lnb_noise_temp_K"
    GS_OTHER_LOSSES_DB = "gs_other_losses_dB"
    
    # Calculated nodes
    GS_ANT_GAIN_DBI = "gs_ant_gain_dBi"
    ANT_NOISE_TEMP_K = "ant_noise_temp_K"
    RX_NOISE_TEMP_K = "rx_noise_temp_K"
    SYS_NOISE_TEMP_K = "sys_noise_temp_K"
    GS_GT_DBK = "gs_gt_dBK"
    CN0_DBHZ = "cn0_dBHz"


@dataclass
class LinkBudgetResult:
    """Container for link budget calculation results."""
    cn0_dBHz: float
    gs_gt_dBK: float
    gs_ant_gain_dBi: float
    sys_noise_temp_K: float
    ant_noise_temp_K: float
    rx_noise_temp_K: float
    link_margin_dB: Optional[float] = None


class LinkBudget:
    """
    Satellite link budget calculator using DAG-based computation.
    
    This class implements a comprehensive link budget that calculates
    the carrier-to-noise-density ratio (C/N0) and related parameters
    for satellite communication links.
    """
    
    def __init__(self, static_params: Dict[str, Any] = None):
        """
        Initialize the link budget calculator.
        
        Args:
            static_params: Dictionary of static link parameters
        """
        logger.info("Initializing LinkBudget calculator...")
        
        # Storage for current parameter values
        self._values = {}
        self._cache = {}  # Cache for calculated values
        
        # Physical constants
        self.BOLTZMANN_CONSTANT_DB = -228.6  # dB(W/K/Hz)
        self.SPEED_OF_LIGHT = 299792458.0    # m/s
        
        # Set static parameters if provided
        if static_params:
            self.update_static_params(static_params)
        
        logger.info("LinkBudget initialized successfully")
    
    def update_static_params(self, params: Dict[str, Any]) -> None:
        """Update static parameters and clear cache."""
        for key, value in params.items():
            self._values[key] = value
        self._cache.clear()
        logger.debug(f"Updated {len(params)} static parameters")
    
    def update_dynamic_params(self, params: Dict[str, Any]) -> None:
        """Update dynamic parameters and clear relevant cache entries."""
        for key, value in params.items():
            self._values[key] = value
        # Clear cache since dynamic parameters changed
        self._cache.clear()
        logger.debug(f"Updated {len(params)} dynamic parameters")
    
    def _get_value(self, node: str) -> Any:
        """Get value for a node, calculating if necessary."""
        if node in self._cache:
            return self._cache[node]
        
        if node in self._values:
            # Direct input value
            value = self._values[node]
        else:
            # Calculate derived value
            value = self._calculate_node(node)
        
        self._cache[node] = value
        return value
    
    def _calculate_node(self, node: str) -> float:
        """Calculate the value of a derived node."""
        if node == LinkBudgetNode.GS_ANT_GAIN_DBI.value:
            return self._calculate_antenna_gain()
        elif node == LinkBudgetNode.ANT_NOISE_TEMP_K.value:
            return self._calculate_antenna_noise_temp()
        elif node == LinkBudgetNode.RX_NOISE_TEMP_K.value:
            return self._calculate_receiver_noise_temp()
        elif node == LinkBudgetNode.SYS_NOISE_TEMP_K.value:
            return self._calculate_system_noise_temp()
        elif node == LinkBudgetNode.GS_GT_DBK.value:
            return self._calculate_gt_ratio()
        elif node == LinkBudgetNode.CN0_DBHZ.value:
            return self._calculate_cn0()
        else:
            raise ValueError(f"Unknown calculation node: {node}")
    
    def _calculate_antenna_gain(self) -> float:
        """Calculate parabolic antenna gain in dBi."""
        freq_GHz = self._get_value(LinkBudgetNode.FREQ_GHZ.value)
        diameter_m = self._get_value(LinkBudgetNode.GS_ANT_DIAM_M.value)
        efficiency = self._get_value(LinkBudgetNode.GS_ANT_EFF.value)
        
        # Standard parabolic antenna gain formula
        wavelength_m = self.SPEED_OF_LIGHT / (freq_GHz * 1e9)
        area_m2 = np.pi * (diameter_m / 2.0) ** 2
        effective_area_m2 = area_m2 * efficiency
        
        gain_linear = (4 * np.pi * effective_area_m2) / (wavelength_m ** 2)
        gain_dBi = 10 * np.log10(gain_linear)
        
        return gain_dBi
    
    def _calculate_antenna_noise_temp(self) -> float:
        """Calculate antenna noise temperature in Kelvin."""
        elevation_deg = self._get_value(LinkBudgetNode.ELEVATION_DEG.value)
        rain_fade_dB = self._get_value(LinkBudgetNode.RAIN_FADE_DB.value)
        
        # Clear sky noise temperature (simplified model)
        # Increases as elevation decreases due to longer atmospheric path
        T_sky_clear = 5.0 + (90.0 - elevation_deg) * 0.5
        
        # Additional noise from rain attenuation
        # Rain acts as a lossy medium at physical temperature
        T_physical_rain = 275.0  # Kelvin, effective rain temperature
        rain_loss_linear = 10 ** (rain_fade_dB / 10.0)
        
        # Noise temperature contribution from rain: T_rain = T_phys * (1 - 1/L)
        if rain_loss_linear > 1.0:
            T_rain = T_physical_rain * (1.0 - 1.0/rain_loss_linear)
        else:
            T_rain = 0.0
        
        return T_sky_clear + T_rain
    
    def _calculate_receiver_noise_temp(self) -> float:
        """Calculate receiver noise temperature in Kelvin."""
        lnb_noise_temp = self._get_value(LinkBudgetNode.GS_LNB_NOISE_TEMP_K.value)
        other_losses_dB = self._get_value(LinkBudgetNode.GS_OTHER_LOSSES_DB.value)
        
        # Noise from passive losses (cables, connectors, etc.)
        T_physical = 290.0  # Room temperature in Kelvin
        loss_linear = 10 ** (other_losses_dB / 10.0)
        
        # Noise from passive losses: T_loss = T_phys * (L - 1)
        if loss_linear > 1.0:
            T_losses = T_physical * (loss_linear - 1.0)
        else:
            T_losses = 0.0
        
        return lnb_noise_temp + T_losses
    
    def _calculate_system_noise_temp(self) -> float:
        """Calculate total system noise temperature in Kelvin."""
        ant_noise_temp = self._get_value(LinkBudgetNode.ANT_NOISE_TEMP_K.value)
        rx_noise_temp = self._get_value(LinkBudgetNode.RX_NOISE_TEMP_K.value)
        
        return ant_noise_temp + rx_noise_temp
    
    def _calculate_gt_ratio(self) -> float:
        """Calculate G/T ratio in dB/K."""
        ant_gain_dBi = self._get_value(LinkBudgetNode.GS_ANT_GAIN_DBI.value)
        sys_noise_temp = self._get_value(LinkBudgetNode.SYS_NOISE_TEMP_K.value)
        
        return ant_gain_dBi - 10 * np.log10(sys_noise_temp)
    
    def _calculate_cn0(self) -> float:
        """Calculate carrier-to-noise-density ratio in dB-Hz."""
        sat_eirp = self._get_value(LinkBudgetNode.SAT_EIRP_DBW.value)
        total_path_loss = self._get_value(LinkBudgetNode.TOTAL_PATH_LOSS_DB.value)
        gs_gt = self._get_value(LinkBudgetNode.GS_GT_DBK.value)
        
        # Link budget equation: C/N0 = EIRP - Path_Loss + G/T - k
        cn0_dBHz = sat_eirp - total_path_loss + gs_gt - self.BOLTZMANN_CONSTANT_DB
        
        return cn0_dBHz
    
    def calculate_link_budget(self, dynamic_params: Dict[str, Any]) -> LinkBudgetResult:
        """
        Calculate complete link budget for given dynamic parameters.
        
        Args:
            dynamic_params: Dictionary with dynamic link parameters
            
        Returns:
            LinkBudgetResult object with all calculated values
        """
        # Update dynamic parameters
        self.update_dynamic_params(dynamic_params)
        
        # Calculate all derived values
        result = LinkBudgetResult(
            cn0_dBHz=self._get_value(LinkBudgetNode.CN0_DBHZ.value),
            gs_gt_dBK=self._get_value(LinkBudgetNode.GS_GT_DBK.value),
            gs_ant_gain_dBi=self._get_value(LinkBudgetNode.GS_ANT_GAIN_DBI.value),
            sys_noise_temp_K=self._get_value(LinkBudgetNode.SYS_NOISE_TEMP_K.value),
            ant_noise_temp_K=self._get_value(LinkBudgetNode.ANT_NOISE_TEMP_K.value),
            rx_noise_temp_K=self._get_value(LinkBudgetNode.RX_NOISE_TEMP_K.value)
        )
        
        return result
    
    def calculate_time_series(self, geometry_df: pd.DataFrame, 
                            loss_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate link budget time series for satellite pass data.
        
        Args:
            geometry_df: DataFrame with geometry data
            loss_df: DataFrame with loss calculations
            
        Returns:
            DataFrame with link budget results for each time step
        """
        logger.info(f"Calculating link budget time series for {len(geometry_df)} time steps")
        
        # Prepare results storage
        results = []
        
        # Calculate for each time step
        for idx, (geom_row, loss_row) in enumerate(zip(geometry_df.itertuples(), 
                                                      loss_df.itertuples())):
            # Dynamic parameters for this time step
            dynamic_params = {
                LinkBudgetNode.ELEVATION_DEG.value: geom_row.elevation_deg,
                LinkBudgetNode.SLANT_RANGE_KM.value: geom_row.slant_range_km,
                LinkBudgetNode.TOTAL_PATH_LOSS_DB.value: loss_row.total_path_loss_dB,
                LinkBudgetNode.RAIN_FADE_DB.value: loss_row.rain_fade_dB
            }
            
            # Calculate link budget
            result = self.calculate_link_budget(dynamic_params)
            
            # Store results
            results.append({
                'timestamp_utc': geom_row.Index,
                'cn0_dBHz': result.cn0_dBHz,
                'gs_gt_dBK': result.gs_gt_dBK,
                'gs_ant_gain_dBi': result.gs_ant_gain_dBi,
                'sys_noise_temp_K': result.sys_noise_temp_K,
                'ant_noise_temp_K': result.ant_noise_temp_K,
                'rx_noise_temp_K': result.rx_noise_temp_K
            })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp_utc', inplace=True)
        
        logger.info("Link budget time series calculation completed")
        return results_df
    
    def analyze_link_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze link performance statistics.
        
        Args:
            results_df: DataFrame with link budget results
            
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'cn0_statistics': {
                'min_dBHz': float(results_df['cn0_dBHz'].min()),
                'max_dBHz': float(results_df['cn0_dBHz'].max()),
                'mean_dBHz': float(results_df['cn0_dBHz'].mean()),
                'std_dBHz': float(results_df['cn0_dBHz'].std()),
                'dynamic_range_dB': float(results_df['cn0_dBHz'].max() - results_df['cn0_dBHz'].min())
            },
            'gt_statistics': {
                'min_dBK': float(results_df['gs_gt_dBK'].min()),
                'max_dBK': float(results_df['gs_gt_dBK'].max()),
                'mean_dBK': float(results_df['gs_gt_dBK'].mean())
            },
            'noise_temp_statistics': {
                'min_sys_K': float(results_df['sys_noise_temp_K'].min()),
                'max_sys_K': float(results_df['sys_noise_temp_K'].max()),
                'mean_sys_K': float(results_df['sys_noise_temp_K'].mean())
            }
        }
        
        logger.info("Link performance analysis completed")
        return stats


def create_default_static_params() -> Dict[str, Any]:
    """
    Create default static parameters for Ku-band satellite link.
    
    Returns:
        Dictionary with default static link parameters
    """
    return {
        LinkBudgetNode.FREQ_GHZ.value: 20.0,                # Ku-band downlink
        LinkBudgetNode.SAT_EIRP_DBW.value: 50.0,            # Typical satellite EIRP
        LinkBudgetNode.GS_ANT_DIAM_M.value: 1.2,            # VSAT antenna diameter
        LinkBudgetNode.GS_ANT_EFF.value: 0.65,              # Antenna efficiency
        LinkBudgetNode.GS_LNB_NOISE_TEMP_K.value: 75.0,     # LNB noise temperature
        LinkBudgetNode.GS_OTHER_LOSSES_DB.value: 0.5        # Cable and other losses
    }


class DVBModcodTable:
    """
    DVB-S2/S2X MODCOD performance tables for adaptive coding and modulation.
    """
    
    def __init__(self):
        """Initialize MODCOD tables."""
        self.dvbs2_table = self._create_dvbs2_table()
        self.dvbs2x_table = self._create_dvbs2x_table()
    
    def _create_dvbs2_table(self) -> pd.DataFrame:
        """Create DVB-S2 MODCOD performance table."""
        # Data from ETSI EN 302 307-1 standard
        modcods = [
            {'name': 'QPSK 1/4', 'eta': 0.490, 'esn0_req_dB': -2.35},
            {'name': 'QPSK 1/3', 'eta': 0.656, 'esn0_req_dB': -1.24},
            {'name': 'QPSK 2/5', 'eta': 0.789, 'esn0_req_dB': -0.30},
            {'name': 'QPSK 1/2', 'eta': 0.989, 'esn0_req_dB': 1.00},
            {'name': 'QPSK 3/5', 'eta': 1.188, 'esn0_req_dB': 2.23},
            {'name': 'QPSK 2/3', 'eta': 1.322, 'esn0_req_dB': 3.10},
            {'name': 'QPSK 3/4', 'eta': 1.487, 'esn0_req_dB': 4.03},
            {'name': 'QPSK 4/5', 'eta': 1.587, 'esn0_req_dB': 4.68},
            {'name': 'QPSK 5/6', 'eta': 1.655, 'esn0_req_dB': 5.18},
            {'name': 'QPSK 8/9', 'eta': 1.766, 'esn0_req_dB': 6.20},
            {'name': 'QPSK 9/10', 'eta': 1.789, 'esn0_req_dB': 6.42},
            {'name': '8PSK 3/5', 'eta': 1.780, 'esn0_req_dB': 5.50},
            {'name': '8PSK 2/3', 'eta': 1.981, 'esn0_req_dB': 6.62},
            {'name': '8PSK 3/4', 'eta': 2.228, 'esn0_req_dB': 7.91},
            {'name': '8PSK 5/6', 'eta': 2.479, 'esn0_req_dB': 9.35},
            {'name': '8PSK 8/9', 'eta': 2.646, 'esn0_req_dB': 10.69},
            {'name': '8PSK 9/10', 'eta': 2.679, 'esn0_req_dB': 10.98},
            {'name': '16APSK 2/3', 'eta': 2.637, 'esn0_req_dB': 8.97},
            {'name': '16APSK 3/4', 'eta': 2.967, 'esn0_req_dB': 10.21},
            {'name': '16APSK 4/5', 'eta': 3.166, 'esn0_req_dB': 11.03},
            {'name': '16APSK 5/6', 'eta': 3.300, 'esn0_req_dB': 11.61},
            {'name': '16APSK 8/9', 'eta': 3.523, 'esn0_req_dB': 12.89},
            {'name': '16APSK 9/10', 'eta': 3.567, 'esn0_req_dB': 13.13},
            {'name': '32APSK 3/4', 'eta': 3.703, 'esn0_req_dB': 12.73},
            {'name': '32APSK 4/5', 'eta': 3.952, 'esn0_req_dB': 13.64},
            {'name': '32APSK 5/6', 'eta': 4.120, 'esn0_req_dB': 14.28},
            {'name': '32APSK 8/9', 'eta': 4.398, 'esn0_req_dB': 15.69},
            {'name': '32APSK 9/10', 'eta': 4.453, 'esn0_req_dB': 16.05}
        ]
        
        df = pd.DataFrame(modcods)
        # Sort by spectral efficiency (descending) for ACM selection
        df = df.sort_values('eta', ascending=False).reset_index(drop=True)
        return df
    
    def _create_dvbs2x_table(self) -> pd.DataFrame:
        """Create selected DVB-S2X MODCOD performance table."""
        # Selected MODCODs from ETSI EN 302 307-2 standard
        modcods = [
            {'name': 'QPSK 2/9', 'eta': 0.435, 'esn0_req_dB': -2.85},
            {'name': 'QPSK 13/45', 'eta': 0.568, 'esn0_req_dB': -2.03},
            {'name': 'QPSK 9/20', 'eta': 0.889, 'esn0_req_dB': 0.22},
            {'name': 'QPSK 11/20', 'eta': 1.089, 'esn0_req_dB': 1.45},
            {'name': '8PSK 23/36', 'eta': 1.896, 'esn0_req_dB': 6.12},
            {'name': '8PSK 13/18', 'eta': 2.145, 'esn0_req_dB': 7.49},
            {'name': '16APSK 26/45', 'eta': 2.282, 'esn0_req_dB': 7.51},
            {'name': '16APSK 7/9', 'eta': 3.077, 'esn0_req_dB': 10.69},
            {'name': '32APSK 2/3-L', 'eta': 3.292, 'esn0_req_dB': 11.10},
            {'name': '32APSK 7/9', 'eta': 3.841, 'esn0_req_dB': 13.05},
            {'name': '64APSK 4/5', 'eta': 4.735, 'esn0_req_dB': 15.87},
            {'name': '64APSK 5/6', 'eta': 4.937, 'esn0_req_dB': 16.55},
            {'name': '128APSK 3/4', 'eta': 5.163, 'esn0_req_dB': 17.73},
            {'name': '256APSK 32/45', 'eta': 5.593, 'esn0_req_dB': 18.59},
            {'name': '256APSK 3/4', 'eta': 5.901, 'esn0_req_dB': 19.57}
        ]
        
        df = pd.DataFrame(modcods)
        df = df.sort_values('eta', ascending=False).reset_index(drop=True)
        return df
    
    def get_combined_table(self) -> pd.DataFrame:
        """Get combined DVB-S2 and DVB-S2X MODCOD table."""
        combined = pd.concat([self.dvbs2_table, self.dvbs2x_table], ignore_index=True)
        return combined.sort_values('eta', ascending=False).reset_index(drop=True) 