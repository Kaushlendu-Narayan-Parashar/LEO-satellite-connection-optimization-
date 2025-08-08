"""
LEO Satellite Communication Simulation Data Generator

This module orchestrates the complete simulation pipeline, combining orbital
mechanics, atmospheric propagation, and link budget calculations to generate
comprehensive datasets for AI training and analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta
from tqdm import tqdm

from .geometry import SatelliteGeometry, create_default_delhi_station
from .propagation import AtmosphericChannel, create_default_link_params, create_default_ground_station_params
from .link_budget import LinkBudget, DVBModcodTable, create_default_static_params, LinkBudgetNode

logger = logging.getLogger(__name__)


class SimulationConfig:
    """Configuration container for simulation parameters."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize with configuration dictionary or defaults."""
        if config_dict is None:
            config_dict = self._get_default_config()
        
        self.satellite_params = config_dict.get('satellite', {})
        self.ground_station_params = config_dict.get('ground_station', {})
        self.link_params = config_dict.get('link', {})
        self.simulation_params = config_dict.get('simulation', {})
        self.data_params = config_dict.get('data_generation', {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default simulation configuration."""
        return {
            'satellite': {
                'tle_url': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
                'tle_filename': 'active.tle',
                'tle_max_age_days': 1.0,
                'target_satellites': ['ISS (ZARYA)', 'STARLINK-1007', 'ONEWEB-0001']
            },
            'ground_station': {
                'name': 'Delhi',
                'lat': 28.61,
                'lon': 77.23,
                'alt_m': 216.0
            },
            'link': {
                'freq_GHz': 20.0,
                'antenna_diameter_m': 1.2,
                'rain_availability_percent': 99.99,
                'sat_eirp_dBW': 50.0,
                'gs_ant_eff': 0.65,
                'gs_lnb_noise_temp_K': 75.0,
                'gs_other_losses_dB': 0.5
            },
            'simulation': {
                'time_step_seconds': 10.0,
                'min_elevation_deg': 5.0,
                'search_duration_hours': 24.0
            },
            'data_generation': {
                'output_dir': 'simulation_data',
                'include_modcod_analysis': True,
                'symbol_rate_Hz': 25e6,
                'implementation_margin_dB': 1.5
            }
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SimulationConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'satellite': self.satellite_params,
            'ground_station': self.ground_station_params,
            'link': self.link_params,
            'simulation': self.simulation_params,
            'data_generation': self.data_params
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


class SimulationDataGenerator:
    """
    Main class for generating LEO satellite communication simulation data.
    
    This class orchestrates the complete simulation pipeline from orbital
    mechanics through link budget calculations to produce comprehensive
    datasets suitable for AI training and analysis.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the simulation data generator.
        
        Args:
            config: Simulation configuration object
        """
        logger.info("Initializing SimulationDataGenerator...")
        
        self.config = config if config else SimulationConfig()
        
        # Initialize simulation components
        self.geometry = SatelliteGeometry()
        self.channel = AtmosphericChannel()
        self.link_budget = LinkBudget()
        self.modcod_table = DVBModcodTable()
        
        # Setup ground station
        gs_params = self.config.ground_station_params
        self.geometry.add_ground_station(
            name=gs_params['name'],
            lat_deg=gs_params['lat'],
            lon_deg=gs_params['lon'],
            alt_m=gs_params['alt_m']
        )
        
        # Setup link budget with static parameters
        static_params = create_default_static_params()
        static_params.update({
            LinkBudgetNode.FREQ_GHZ.value: self.config.link_params['freq_GHz'],
            LinkBudgetNode.SAT_EIRP_DBW.value: self.config.link_params['sat_eirp_dBW'],
            LinkBudgetNode.GS_ANT_DIAM_M.value: self.config.link_params['antenna_diameter_m'],
            LinkBudgetNode.GS_ANT_EFF.value: self.config.link_params['gs_ant_eff'],
            LinkBudgetNode.GS_LNB_NOISE_TEMP_K.value: self.config.link_params['gs_lnb_noise_temp_K'],
            LinkBudgetNode.GS_OTHER_LOSSES_DB.value: self.config.link_params['gs_other_losses_dB']
        })
        self.link_budget.update_static_params(static_params)
        
        logger.info("SimulationDataGenerator initialized successfully")
    
    def load_satellite_data(self) -> Dict[str, Any]:
        """Load TLE data for satellites."""
        sat_params = self.config.satellite_params
        return self.geometry.load_tle_data(
            tle_url=sat_params.get('tle_url'),
            filename=sat_params.get('tle_filename', 'active.tle'),
            max_age_days=sat_params.get('tle_max_age_days', 1.0)
        )
    
    def simulate_satellite_pass(self, satellite_name: str) -> Optional[pd.DataFrame]:
        """
        Simulate a complete satellite pass with all link calculations.
        
        Args:
            satellite_name: Name of satellite to simulate
            
        Returns:
            DataFrame with complete simulation data, or None if no pass found
        """
        logger.info(f"Simulating pass for satellite: {satellite_name}")
        
        # Get next pass geometry
        geometry_df = self.geometry.get_next_pass_geometry(
            satellite_name=satellite_name,
            ground_station_name=self.config.ground_station_params['name'],
            time_step_seconds=self.config.simulation_params['time_step_seconds'],
            min_elevation_deg=self.config.simulation_params['min_elevation_deg']
        )
        
        if geometry_df is None:
            logger.warning(f"No pass found for {satellite_name}")
            return None
        
        logger.info(f"Found pass with {len(geometry_df)} time steps")
        
        # Calculate atmospheric losses
        gs_params = self.config.ground_station_params
        link_params = {
            'freq_GHz': self.config.link_params['freq_GHz'],
            'antenna_diameter_m': self.config.link_params['antenna_diameter_m'],
            'rain_availability_percent': self.config.link_params['rain_availability_percent']
        }
        
        loss_df = self.channel.calculate_total_path_loss(
            geometry_df, gs_params, link_params
        )
        
        # Calculate link budget time series
        link_results_df = self.link_budget.calculate_time_series(geometry_df, loss_df)
        
        # Combine all data
        simulation_df = geometry_df.join([loss_df, link_results_df])
        
        # Add MODCOD analysis if requested
        if self.config.data_params.get('include_modcod_analysis', True):
            modcod_df = self._add_modcod_analysis(simulation_df)
            simulation_df = simulation_df.join(modcod_df)
        
        logger.info(f"Simulation completed for {satellite_name}")
        return simulation_df
    
    def _add_modcod_analysis(self, simulation_df: pd.DataFrame) -> pd.DataFrame:
        """Add MODCOD selection analysis to simulation data."""
        logger.info("Adding MODCOD analysis...")
        
        symbol_rate = self.config.data_params.get('symbol_rate_Hz', 25e6)
        impl_margin = self.config.data_params.get('implementation_margin_dB', 1.5)
        
        # Get combined MODCOD table
        modcod_table = self.modcod_table.get_combined_table()
        
        results = []
        
        for idx, row in simulation_df.iterrows():
            cn0_available = row['cn0_dBHz']
            
            # Convert C/N0 to Es/N0
            esn0_available = cn0_available - 10 * np.log10(symbol_rate)
            
            # Find best MODCOD
            selected_modcod = None
            data_rate_mbps = 0.0
            link_margin_db = -999.0
            
            for _, modcod in modcod_table.iterrows():
                required_esn0 = modcod['esn0_req_dB'] + impl_margin
                
                if esn0_available >= required_esn0:
                    selected_modcod = modcod['name']
                    data_rate_mbps = (symbol_rate * modcod['eta']) / 1e6
                    link_margin_db = esn0_available - required_esn0
                    break
            
            if selected_modcod is None:
                selected_modcod = "Link Down"
                data_rate_mbps = 0.0
                link_margin_db = esn0_available - modcod_table.iloc[-1]['esn0_req_dB']
            
            results.append({
                'esn0_available_dB': esn0_available,
                'selected_modcod': selected_modcod,
                'data_rate_mbps': data_rate_mbps,
                'link_margin_dB': link_margin_db
            })
        
        modcod_df = pd.DataFrame(results, index=simulation_df.index)
        logger.info("MODCOD analysis completed")
        return modcod_df
    
    def generate_multi_satellite_dataset(self, satellite_names: List[str] = None,
                                       max_passes_per_satellite: int = 5) -> pd.DataFrame:
        """
        Generate dataset with multiple satellite passes.
        
        Args:
            satellite_names: List of satellite names to simulate
            max_passes_per_satellite: Maximum passes to simulate per satellite
            
        Returns:
            Combined DataFrame with all simulation data
        """
        if satellite_names is None:
            satellite_names = self.config.satellite_params.get('target_satellites', ['ISS (ZARYA)'])
        
        logger.info(f"Generating multi-satellite dataset for {len(satellite_names)} satellites")
        
        # Load satellite data
        satellites = self.load_satellite_data()
        
        all_passes = []
        
        for sat_name in tqdm(satellite_names, desc="Satellites"):
            if sat_name not in satellites:
                logger.warning(f"Satellite {sat_name} not found in TLE data")
                continue
            
            passes_generated = 0
            
            # Try to generate multiple passes (by advancing time)
            for pass_attempt in range(max_passes_per_satellite):
                try:
                    pass_df = self.simulate_satellite_pass(sat_name)
                    
                    if pass_df is not None:
                        # Add metadata
                        pass_df['satellite_name'] = sat_name
                        pass_df['pass_number'] = passes_generated + 1
                        all_passes.append(pass_df)
                        passes_generated += 1
                        
                        logger.info(f"Generated pass {passes_generated} for {sat_name}")
                    
                    if passes_generated >= max_passes_per_satellite:
                        break
                        
                except Exception as e:
                    logger.error(f"Error simulating {sat_name} pass {pass_attempt}: {e}")
                    continue
        
        if not all_passes:
            logger.error("No passes generated for any satellite")
            return pd.DataFrame()
        
        # Combine all passes
        combined_df = pd.concat(all_passes, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Generated dataset with {len(combined_df)} total time steps from {len(all_passes)} passes")
        return combined_df
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str = None, 
                    filename_prefix: str = "leo_simulation") -> Dict[str, str]:
        """
        Save simulation dataset in multiple formats.
        
        Args:
            df: DataFrame to save
            output_dir: Output directory path
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if output_dir is None:
            output_dir = self.config.data_params.get('output_dir', 'simulation_data')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        saved_files = {}
        
        # Save as CSV
        csv_path = output_path / f"{base_filename}.csv"
        df.to_csv(csv_path)
        saved_files['csv'] = str(csv_path)
        
        # Save as Parquet (more efficient for large datasets)
        try:
            parquet_path = output_path / f"{base_filename}.parquet"
            df.to_parquet(parquet_path)
            saved_files['parquet'] = str(parquet_path)
        except ImportError:
            logger.warning("Parquet support not available, skipping parquet export")
        
        # Save metadata
        metadata = {
            'generation_timestamp': timestamp,
            'num_records': len(df),
            'num_satellites': df['satellite_name'].nunique() if 'satellite_name' in df.columns else 1,
            'time_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'columns': list(df.columns),
            'config': {
                'satellite': self.config.satellite_params,
                'ground_station': self.config.ground_station_params,
                'link': self.config.link_params,
                'simulation': self.config.simulation_params
            }
        }
        
        metadata_path = output_path / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = str(metadata_path)
        
        logger.info(f"Dataset saved to {output_dir} in formats: {list(saved_files.keys())}")
        return saved_files
    
    def generate_analysis_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report of simulation data.
        
        Args:
            df: Simulation DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Generating analysis report...")
        
        report = {
            'dataset_summary': {
                'total_records': len(df),
                'time_span_hours': (df.index.max() - df.index.min()).total_seconds() / 3600 if len(df) > 0 else 0,
                'satellites': df['satellite_name'].unique().tolist() if 'satellite_name' in df.columns else ['Unknown'],
                'passes': df['pass_number'].nunique() if 'pass_number' in df.columns else 1
            },
            'link_performance': {
                'cn0_range_dB': {
                    'min': float(df['cn0_dBHz'].min()),
                    'max': float(df['cn0_dBHz'].max()),
                    'mean': float(df['cn0_dBHz'].mean()),
                    'std': float(df['cn0_dBHz'].std())
                },
                'elevation_range': {
                    'min': float(df['elevation_deg'].min()),
                    'max': float(df['elevation_deg'].max()),
                    'mean': float(df['elevation_deg'].mean())
                }
            }
        }
        
        # Add MODCOD analysis if available
        if 'selected_modcod' in df.columns:
            modcod_counts = df['selected_modcod'].value_counts()
            report['modcod_analysis'] = {
                'modcod_distribution': modcod_counts.to_dict(),
                'average_data_rate_mbps': float(df[df['data_rate_mbps'] > 0]['data_rate_mbps'].mean()),
                'link_availability_percent': float((df['selected_modcod'] != 'Link Down').mean() * 100)
            }
        
        # Add atmospheric loss analysis
        if 'total_atmospheric_loss_dB' in df.columns:
            report['atmospheric_analysis'] = {
                'total_atmospheric_loss': {
                    'min': float(df['total_atmospheric_loss_dB'].min()),
                    'max': float(df['total_atmospheric_loss_dB'].max()),
                    'mean': float(df['total_atmospheric_loss_dB'].mean())
                },
                'rain_fade': {
                    'min': float(df['rain_fade_dB'].min()),
                    'max': float(df['rain_fade_dB'].max()),
                    'mean': float(df['rain_fade_dB'].mean())
                }
            }
        
        logger.info("Analysis report generated")
        return report


def create_example_simulation() -> Tuple[SimulationDataGenerator, pd.DataFrame]:
    """
    Create an example simulation for demonstration purposes.
    
    Returns:
        Tuple of (generator instance, simulation DataFrame)
    """
    logger.info("Creating example simulation...")
    
    # Create generator with default configuration
    generator = SimulationDataGenerator()
    
    # Generate dataset for ISS
    df = generator.simulate_satellite_pass('ISS (ZARYA)')
    
    if df is None:
        logger.error("Failed to generate example simulation")
        return generator, pd.DataFrame()
    
    logger.info(f"Example simulation created with {len(df)} records")
    return generator, df 