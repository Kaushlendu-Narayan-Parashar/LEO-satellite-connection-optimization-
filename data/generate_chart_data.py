#!/usr/bin/env python3
"""
LEO Satellite Data Generation for Chart Visualization

This script generates a moderate amount of high-quality data specifically
optimized for creating beautiful charts and visualizations.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from leo_simulation import SimulationDataGenerator
from leo_simulation.data_generator import SimulationConfig

def generate_chart_optimized_data():
    """Generate data optimized for chart visualization."""
    print("🎯 GENERATING CHART-OPTIMIZED DATASET")
    print("="*60)
    
    start_time = time.time()
    
    # Create generator
    generator = SimulationDataGenerator()
    
    # Load satellites
    print("📡 Loading satellite catalog...")
    satellites = generator.load_satellite_data()
    print(f"✅ Available satellites: {len(satellites):,}")
    
    # Select diverse, well-known satellites for interesting charts
    target_satellites = [
        'ISS (ZARYA)',           # Space station - low orbit, high speed
        'NOAA 18',               # Weather satellite - polar orbit
        'NOAA 19',               # Weather satellite - polar orbit  
        'TERRA',                 # Earth observation - sun-synchronous
        'AQUA',                  # Earth observation - sun-synchronous
        'LANDSAT 8',             # Earth observation - different orbit
        'GOES-16',               # Geostationary weather satellite
        'SENTINEL-1A',           # European Earth observation
    ]
    
    # Find available satellites
    available_sats = [sat for sat in target_satellites if sat in satellites]
    
    # Add more if needed
    if len(available_sats) < 6:
        # Add some common satellite types
        all_names = list(satellites.keys())
        additional = [sat for sat in all_names if sat not in available_sats 
                     and any(keyword in sat.upper() for keyword in ['NOAA', 'TERRA', 'AQUA', 'LANDSAT'])]
        available_sats.extend(additional[:6-len(available_sats)])
    
    selected_satellites = available_sats[:6]  # Use 6 satellites for good variety
    
    print(f"🛰️  Selected satellites for visualization:")
    for i, sat in enumerate(selected_satellites, 1):
        print(f"   {i}. {sat}")
    
    # Generate dataset with multiple passes per satellite
    print(f"\n⚡ Generating data (2-3 passes per satellite)...")
    generation_start = time.time()
    
    dataset = generator.generate_multi_satellite_dataset(
        satellite_names=selected_satellites,
        max_passes_per_satellite=3  # 3 passes each for variety
    )
    
    generation_time = time.time() - generation_start
    
    if dataset.empty:
        print("❌ No data generated")
        return None
    
    # Add some derived columns that are useful for charts
    print("📊 Adding chart-friendly derived columns...")
    
    # Add time-based columns
    dataset['hour_utc'] = dataset.index.hour
    dataset['day_of_year'] = dataset.index.dayofyear
    
    # Add categorical elevation ranges
    dataset['elevation_range'] = pd.cut(dataset['elevation_deg'], 
                                       bins=[0, 10, 30, 60, 90], 
                                       labels=['Low (5-10°)', 'Medium (10-30°)', 'High (30-60°)', 'Very High (60-90°)'])
    
    # Add signal quality categories
    dataset['signal_quality'] = pd.cut(dataset['cn0_dBHz'], 
                                      bins=[-np.inf, 40, 60, 80, np.inf],
                                      labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    # Add satellite type categories (simplified)
    def categorize_satellite(name):
        if 'ISS' in name:
            return 'Space Station'
        elif 'NOAA' in name:
            return 'Weather'
        elif any(x in name for x in ['TERRA', 'AQUA', 'LANDSAT', 'SENTINEL']):
            return 'Earth Observation'
        elif 'GOES' in name:
            return 'Geostationary'
        else:
            return 'Other'
    
    dataset['satellite_type'] = dataset['satellite_name'].apply(categorize_satellite)
    
    # Save the dataset
    print(f"\n💾 Saving chart-optimized dataset...")
    save_start = time.time()
    saved_files = generator.save_dataset(dataset, filename_prefix="chart_data")
    save_time = time.time() - save_start
    
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n📈 CHART DATA GENERATION RESULTS")
    print("="*60)
    
    unique_satellites = dataset['satellite_name'].nunique()
    unique_passes = dataset['pass_number'].nunique() if 'pass_number' in dataset.columns else 0
    time_span_hours = (dataset.index.max() - dataset.index.min()).total_seconds() / 3600
    
    print(f"📊 DATASET SUMMARY:")
    print(f"   • Total data points: {len(dataset):,}")
    print(f"   • Satellites: {unique_satellites}")
    print(f"   • Total passes: {unique_passes}")
    print(f"   • Time span: {time_span_hours:.1f} hours")
    print(f"   • Columns: {len(dataset.columns)}")
    print(f"   • Generation time: {total_time:.1f} seconds")
    
    print(f"\n🎯 CHART-READY FEATURES:")
    print(f"   • Elevation ranges: {dataset['elevation_range'].value_counts().to_dict()}")
    print(f"   • Signal quality: {dataset['signal_quality'].value_counts().to_dict()}")
    print(f"   • Satellite types: {dataset['satellite_type'].value_counts().to_dict()}")
    
    if 'selected_modcod' in dataset.columns:
        link_up_pct = (dataset['selected_modcod'] != 'Link Down').mean() * 100
        print(f"   • Link availability: {link_up_pct:.1f}%")
    
    print(f"\n💾 OUTPUT FILES:")
    for format_type, file_path in saved_files.items():
        if format_type == 'csv':
            file_size_mb = Path(file_path).stat().st_size / (1024*1024)
            print(f"   • {format_type}: {file_path} ({file_size_mb:.1f} MB)")
        else:
            print(f"   • {format_type}: {file_path}")
    
    return dataset, saved_files

def main():
    """Main function to generate chart-optimized data."""
    print("📊 LEO SATELLITE CHART DATA GENERATOR")
    print("="*60)
    print("Generating moderate dataset optimized for beautiful charts")
    
    try:
        dataset, files = generate_chart_optimized_data()
        
        if dataset is not None:
            print(f"\n✅ SUCCESS!")
            print(f"📁 Your chart-ready dataset is saved in:")
            print(f"   {files['csv']}")
            print(f"\n🚀 Next step: Run the chart generation script!")
            print(f"   python3 create_charts.py")
        
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        raise

if __name__ == "__main__":
    main() 