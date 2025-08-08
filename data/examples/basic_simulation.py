#!/usr/bin/env python3
"""
Basic LEO Satellite Communication Simulation Example

This script demonstrates how to use the LEO simulation framework to generate
satellite communication link data for AI training and analysis.
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to Python path to import leo_simulation
sys.path.insert(0, str(Path(__file__).parent.parent))

from leo_simulation import SimulationDataGenerator, SimulationConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_basic_simulation():
    """Run a basic simulation example."""
    logger.info("Starting basic LEO satellite simulation...")
    
    try:
        # Create simulation generator with default configuration
        generator = SimulationDataGenerator()
        
        # Load satellite TLE data
        satellites = generator.load_satellite_data()
        logger.info(f"Loaded {len(satellites)} satellites")
        
        # List available satellites (first 10)
        sat_names = list(satellites.keys())[:10]
        logger.info(f"Available satellites: {sat_names}")
        
        # Simulate ISS pass
        logger.info("Simulating ISS pass...")
        iss_data = generator.simulate_satellite_pass('ISS (ZARYA)')
        
        if iss_data is None:
            logger.error("No ISS pass found in the next 24 hours")
            return None
        
        logger.info(f"Generated {len(iss_data)} data points for ISS pass")
        
        # Display basic statistics
        print("\n" + "="*60)
        print("BASIC SIMULATION RESULTS")
        print("="*60)
        
        print(f"Pass duration: {len(iss_data) * 10 / 60:.1f} minutes")
        print(f"Max elevation: {iss_data['elevation_deg'].max():.1f}°")
        print(f"C/N0 range: {iss_data['cn0_dBHz'].min():.1f} to {iss_data['cn0_dBHz'].max():.1f} dB-Hz")
        
        if 'selected_modcod' in iss_data.columns:
            modcod_counts = iss_data['selected_modcod'].value_counts()
            print(f"\nMODCOD distribution:")
            for modcod, count in modcod_counts.head().items():
                percentage = (count / len(iss_data)) * 100
                print(f"  {modcod}: {percentage:.1f}%")
        
        # Save data
        saved_files = generator.save_dataset(iss_data, filename_prefix="basic_iss_example")
        logger.info(f"Data saved: {saved_files}")
        
        # Generate analysis report
        report = generator.generate_analysis_report(iss_data)
        print(f"\nAnalysis report generated with {len(report)} sections")
        
        return iss_data
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def create_visualizations(data: pd.DataFrame):
    """Create basic visualizations of the simulation data."""
    if data is None or data.empty:
        logger.warning("No data available for visualization")
        return
    
    logger.info("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LEO Satellite Communication Link Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Elevation vs C/N0
    axes[0, 0].scatter(data['elevation_deg'], data['cn0_dBHz'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Elevation Angle (degrees)')
    axes[0, 0].set_ylabel('C/N0 (dB-Hz)')
    axes[0, 0].set_title('Link Quality vs Elevation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time series of key parameters
    time_minutes = (data.index - data.index[0]).total_seconds() / 60
    axes[0, 1].plot(time_minutes, data['elevation_deg'], label='Elevation (deg)', linewidth=2)
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Elevation (degrees)', color='blue')
    axes[0, 1].tick_params(axis='y', labelcolor='blue')
    
    ax2 = axes[0, 1].twinx()
    ax2.plot(time_minutes, data['cn0_dBHz'], label='C/N0 (dB-Hz)', color='red', linewidth=2)
    ax2.set_ylabel('C/N0 (dB-Hz)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    axes[0, 1].set_title('Pass Timeline')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss breakdown
    if 'total_path_loss_dB' in data.columns:
        axes[1, 0].plot(time_minutes, data['fspl_dB'], label='FSPL', linewidth=2)
        axes[1, 0].plot(time_minutes, data['total_atmospheric_loss_dB'], label='Atmospheric', linewidth=2)
        axes[1, 0].plot(time_minutes, data['total_path_loss_dB'], label='Total', linewidth=2, linestyle='--')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Path Loss (dB)')
        axes[1, 0].set_title('Path Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: MODCOD distribution (if available)
    if 'selected_modcod' in data.columns:
        modcod_counts = data['selected_modcod'].value_counts()
        if len(modcod_counts) > 0:
            # Only show top 8 MODCODs for readability
            top_modcods = modcod_counts.head(8)
            axes[1, 1].pie(top_modcods.values, labels=top_modcods.index, autopct='%1.1f%%')
            axes[1, 1].set_title('MODCOD Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'MODCOD analysis\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('MODCOD Analysis')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('simulation_data')
    output_dir.mkdir(exist_ok=True)
    plot_filename = output_dir / 'basic_simulation_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {plot_filename}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        logger.info("Plot display not available (non-interactive environment)")


def run_multi_satellite_example():
    """Run example with multiple satellites."""
    logger.info("Running multi-satellite example...")
    
    try:
        # Create configuration for multiple satellites
        config = SimulationConfig()
        config.satellite_params['target_satellites'] = ['ISS (ZARYA)', 'NOAA 18', 'NOAA 19']
        
        generator = SimulationDataGenerator(config)
        
        # Generate dataset with multiple satellites (1 pass each for speed)
        multi_sat_data = generator.generate_multi_satellite_dataset(
            satellite_names=['ISS (ZARYA)', 'NOAA 18', 'NOAA 19'],
            max_passes_per_satellite=1
        )
        
        if multi_sat_data.empty:
            logger.warning("No multi-satellite data generated")
            return None
        
        logger.info(f"Generated {len(multi_sat_data)} total data points")
        
        # Display summary by satellite
        print("\n" + "="*60)
        print("MULTI-SATELLITE SIMULATION RESULTS")
        print("="*60)
        
        if 'satellite_name' in multi_sat_data.columns:
            for sat_name in multi_sat_data['satellite_name'].unique():
                sat_data = multi_sat_data[multi_sat_data['satellite_name'] == sat_name]
                print(f"\n{sat_name}:")
                print(f"  Data points: {len(sat_data)}")
                print(f"  Max elevation: {sat_data['elevation_deg'].max():.1f}°")
                print(f"  C/N0 range: {sat_data['cn0_dBHz'].min():.1f} to {sat_data['cn0_dBHz'].max():.1f} dB-Hz")
        
        # Save multi-satellite dataset
        saved_files = generator.save_dataset(multi_sat_data, filename_prefix="multi_satellite_example")
        logger.info(f"Multi-satellite data saved: {saved_files}")
        
        return multi_sat_data
        
    except Exception as e:
        logger.error(f"Multi-satellite simulation failed: {e}")
        return None


def main():
    """Main function to run all examples."""
    print("LEO Satellite Communication Simulation Framework")
    print("=" * 50)
    
    # Run basic simulation
    basic_data = run_basic_simulation()
    
    if basic_data is not None:
        # Create visualizations
        try:
            create_visualizations(basic_data)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # Run multi-satellite example
    multi_data = run_multi_satellite_example()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("Check the 'simulation_data' directory for output files.")
    print("Generated data can be used for AI training and analysis.")


if __name__ == "__main__":
    main() 