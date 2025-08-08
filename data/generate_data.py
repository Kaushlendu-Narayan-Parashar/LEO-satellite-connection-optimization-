#!/usr/bin/env python3
"""
LEO Satellite Communication Data Generation Script

This script demonstrates how to generate different amounts of simulation data
and shows the system's capacity and limitations.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from leo_simulation import SimulationDataGenerator
from leo_simulation.data_generator import SimulationConfig

def generate_single_pass_data():
    """Generate data for a single satellite pass."""
    print("\n" + "="*60)
    print("🛰️  SINGLE SATELLITE PASS GENERATION")
    print("="*60)
    
    start_time = time.time()
    
    # Create generator
    generator = SimulationDataGenerator()
    
    # Load satellite data
    print("📡 Loading satellite TLE data...")
    satellites = generator.load_satellite_data()
    print(f"✅ Loaded {len(satellites)} satellites")
    
    # Generate single pass
    print("🚀 Generating ISS pass data...")
    iss_data = generator.simulate_satellite_pass('ISS (ZARYA)')
    
    if iss_data is None:
        print("❌ No ISS pass found in next 24 hours")
        return None
    
    # Save data
    saved_files = generator.save_dataset(iss_data, filename_prefix="single_pass")
    
    # Statistics
    duration = time.time() - start_time
    print(f"\n📊 SINGLE PASS RESULTS:")
    print(f"   • Data points: {len(iss_data):,}")
    print(f"   • Columns: {len(iss_data.columns)}")
    print(f"   • Pass duration: {len(iss_data) * 10 / 60:.1f} minutes")
    print(f"   • Max elevation: {iss_data['elevation_deg'].max():.1f}°")
    print(f"   • C/N0 range: {iss_data['cn0_dBHz'].min():.1f} to {iss_data['cn0_dBHz'].max():.1f} dB-Hz")
    print(f"   • Generation time: {duration:.1f} seconds")
    print(f"   • Data rate: {len(iss_data)/duration:.0f} points/second")
    
    if 'selected_modcod' in iss_data.columns:
        modcods = iss_data['selected_modcod'].value_counts()
        print(f"   • Top MODCOD: {modcods.index[0]} ({modcods.iloc[0]/len(iss_data)*100:.1f}%)")
    
    print(f"💾 Files saved: {list(saved_files.keys())}")
    
    return iss_data

def generate_multi_satellite_data(num_satellites=5, passes_per_sat=2):
    """Generate data for multiple satellites."""
    print("\n" + "="*60)
    print(f"🌍 MULTI-SATELLITE DATA GENERATION")
    print(f"    {num_satellites} satellites × {passes_per_sat} passes each")
    print("="*60)
    
    start_time = time.time()
    
    # Create generator
    generator = SimulationDataGenerator()
    
    # Load satellites
    satellites = generator.load_satellite_data()
    
    # Select satellites (mix of different types)
    satellite_candidates = [
        'ISS (ZARYA)',
        'NOAA 18', 'NOAA 19', 'NOAA 20',
        'TERRA', 'AQUA',
        'LANDSAT 8', 'LANDSAT 9',
        'SENTINEL-1A', 'SENTINEL-1B',
        'SENTINEL-2A', 'SENTINEL-2B',
        'SENTINEL-3A', 'SENTINEL-3B'
    ]
    
    # Find available satellites
    available_sats = [sat for sat in satellite_candidates if sat in satellites][:num_satellites]
    
    if len(available_sats) < num_satellites:
        # Fill with any available satellites
        all_sat_names = list(satellites.keys())
        additional_sats = [sat for sat in all_sat_names if sat not in available_sats][:num_satellites - len(available_sats)]
        available_sats.extend(additional_sats)
    
    print(f"🛰️  Selected satellites: {available_sats}")
    
    # Generate multi-satellite dataset
    multi_data = generator.generate_multi_satellite_dataset(
        satellite_names=available_sats,
        max_passes_per_satellite=passes_per_sat
    )
    
    if multi_data.empty:
        print("❌ No multi-satellite data generated")
        return None
    
    # Save data
    saved_files = generator.save_dataset(multi_data, filename_prefix="multi_satellite")
    
    # Generate analysis report
    report = generator.generate_analysis_report(multi_data)
    
    # Statistics
    duration = time.time() - start_time
    unique_sats = multi_data['satellite_name'].nunique() if 'satellite_name' in multi_data.columns else 0
    unique_passes = multi_data['pass_number'].nunique() if 'pass_number' in multi_data.columns else 0
    
    print(f"\n📊 MULTI-SATELLITE RESULTS:")
    print(f"   • Total data points: {len(multi_data):,}")
    print(f"   • Unique satellites: {unique_sats}")
    print(f"   • Total passes: {unique_passes}")
    print(f"   • Columns: {len(multi_data.columns)}")
    print(f"   • Time span: {(multi_data.index.max() - multi_data.index.min()).total_seconds() / 3600:.1f} hours")
    print(f"   • Generation time: {duration:.1f} seconds")
    print(f"   • Data rate: {len(multi_data)/duration:.0f} points/second")
    
    # Per-satellite breakdown
    if 'satellite_name' in multi_data.columns:
        print(f"\n📈 PER-SATELLITE BREAKDOWN:")
        for sat_name in multi_data['satellite_name'].unique():
            sat_data = multi_data[multi_data['satellite_name'] == sat_name]
            passes = sat_data['pass_number'].nunique() if 'pass_number' in sat_data.columns else 1
            print(f"   • {sat_name}: {len(sat_data):,} points ({passes} passes)")
    
    print(f"💾 Files saved: {list(saved_files.keys())}")
    
    return multi_data

def estimate_system_capacity():
    """Estimate the system's data generation capacity."""
    print("\n" + "="*60)
    print("⚡ SYSTEM CAPACITY ANALYSIS")
    print("="*60)
    
    # Generate small sample to estimate performance
    print("🔍 Running capacity test...")
    start_time = time.time()
    
    generator = SimulationDataGenerator()
    satellites = generator.load_satellite_data()
    
    # Test with ISS (usually available)
    test_data = generator.simulate_satellite_pass('ISS (ZARYA)')
    
    if test_data is None:
        print("❌ Cannot estimate capacity - no test pass available")
        return
    
    test_duration = time.time() - start_time
    points_per_second = len(test_data) / test_duration
    
    print(f"\n📊 CAPACITY ESTIMATES:")
    print(f"   • Test pass: {len(test_data)} points in {test_duration:.1f}s")
    print(f"   • Processing rate: {points_per_second:.0f} points/second")
    
    # Typical pass characteristics
    typical_pass_duration_min = 10  # minutes
    typical_points_per_pass = typical_pass_duration_min * 60 / 10  # 10-second intervals
    time_per_pass = typical_points_per_pass / points_per_second
    
    print(f"\n🎯 GENERATION CAPACITY:")
    print(f"   • Typical pass: ~{typical_points_per_pass:.0f} points ({typical_pass_duration_min} min)")
    print(f"   • Time per pass: ~{time_per_pass:.1f} seconds")
    
    # Hourly/daily estimates
    passes_per_hour = 3600 / time_per_pass
    passes_per_day = passes_per_hour * 24
    points_per_hour = passes_per_hour * typical_points_per_pass
    points_per_day = points_per_hour * 24
    
    print(f"\n⏱️  THROUGHPUT ESTIMATES:")
    print(f"   • Passes per hour: ~{passes_per_hour:.0f}")
    print(f"   • Passes per day: ~{passes_per_day:.0f}")
    print(f"   • Points per hour: ~{points_per_hour:,.0f}")
    print(f"   • Points per day: ~{points_per_day:,.0f}")
    
    # Storage estimates
    bytes_per_point = test_data.memory_usage(deep=True).sum() / len(test_data)
    mb_per_day = points_per_day * bytes_per_point / (1024*1024)
    gb_per_day = mb_per_day / 1024
    
    print(f"\n💾 STORAGE ESTIMATES:")
    print(f"   • Bytes per data point: ~{bytes_per_point:.0f}")
    print(f"   • Storage per day: ~{mb_per_day:.0f} MB (~{gb_per_day:.1f} GB)")
    
    # Practical limits
    print(f"\n🚀 PRACTICAL GENERATION SCENARIOS:")
    
    scenarios = [
        ("Small dataset", 10, 1, "Research/Testing"),
        ("Medium dataset", 50, 2, "Model training"),
        ("Large dataset", 100, 5, "Production AI training"),
        ("Massive dataset", 500, 10, "Comprehensive analysis")
    ]
    
    for name, sats, passes, use_case in scenarios:
        total_passes = sats * passes
        total_points = total_passes * typical_points_per_pass
        total_time_hours = total_passes * time_per_pass / 3600
        storage_mb = total_points * bytes_per_point / (1024*1024)
        
        print(f"   • {name}:")
        print(f"     - {sats} satellites × {passes} passes = {total_passes} passes")
        print(f"     - ~{total_points:,.0f} data points")
        print(f"     - Generation time: ~{total_time_hours:.1f} hours")
        print(f"     - Storage: ~{storage_mb:.0f} MB")
        print(f"     - Use case: {use_case}")

def demonstrate_data_features():
    """Show what features are available in the generated data."""
    print("\n" + "="*60)
    print("📋 DATA FEATURES DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    generator = SimulationDataGenerator()
    satellites = generator.load_satellite_data()
    sample_data = generator.simulate_satellite_pass('ISS (ZARYA)')
    
    if sample_data is None:
        print("❌ Cannot demonstrate features - no sample data")
        return
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Shape: {sample_data.shape}")
    print(f"   • Memory usage: {sample_data.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    print(f"   • Time range: {sample_data.index[0]} to {sample_data.index[-1]}")
    
    print(f"\n🏗️  COLUMN CATEGORIES:")
    
    # Categorize columns
    geometry_cols = [col for col in sample_data.columns if any(x in col.lower() for x in ['elevation', 'azimuth', 'range', 'satellite', 'ground'])]
    loss_cols = [col for col in sample_data.columns if any(x in col.lower() for x in ['loss', 'fade', 'fspl', 'atmospheric'])]
    link_cols = [col for col in sample_data.columns if any(x in col.lower() for x in ['cn0', 'gt', 'gain', 'noise', 'temp'])]
    modcod_cols = [col for col in sample_data.columns if any(x in col.lower() for x in ['modcod', 'esn0', 'data_rate', 'margin'])]
    
    print(f"   • Geometry ({len(geometry_cols)}): {geometry_cols[:3]}{'...' if len(geometry_cols) > 3 else ''}")
    print(f"   • Path Losses ({len(loss_cols)}): {loss_cols[:3]}{'...' if len(loss_cols) > 3 else ''}")
    print(f"   • Link Budget ({len(link_cols)}): {link_cols[:3]}{'...' if len(link_cols) > 3 else ''}")
    print(f"   • MODCOD/ACM ({len(modcod_cols)}): {modcod_cols[:3]}{'...' if len(modcod_cols) > 3 else ''}")
    
    print(f"\n📈 SAMPLE DATA RANGES:")
    key_columns = ['elevation_deg', 'slant_range_km', 'total_path_loss_dB', 'cn0_dBHz']
    for col in key_columns:
        if col in sample_data.columns:
            print(f"   • {col}: {sample_data[col].min():.2f} to {sample_data[col].max():.2f}")
    
    if 'selected_modcod' in sample_data.columns:
        modcod_dist = sample_data['selected_modcod'].value_counts()
        print(f"\n🎛️  MODCOD DISTRIBUTION:")
        for modcod, count in modcod_dist.head(5).items():
            pct = count / len(sample_data) * 100
            print(f"   • {modcod}: {count} points ({pct:.1f}%)")
    
    # Show sample rows
    print(f"\n📋 SAMPLE DATA (first 3 rows):")
    display_cols = ['elevation_deg', 'cn0_dBHz', 'selected_modcod', 'data_rate_mbps'] if 'selected_modcod' in sample_data.columns else ['elevation_deg', 'cn0_dBHz']
    print(sample_data[display_cols].head(3).to_string())

def main():
    """Main function to demonstrate data generation capabilities."""
    print("🛰️  LEO SATELLITE COMMUNICATION DATA GENERATOR")
    print("=" * 60)
    print("This script demonstrates the system's data generation capabilities")
    print("and shows how much data you can generate.")
    
    try:
        # 1. Single pass generation
        single_data = generate_single_pass_data()
        
        # 2. Multi-satellite generation
        multi_data = generate_multi_satellite_data(num_satellites=3, passes_per_sat=1)
        
        # 3. System capacity analysis
        estimate_system_capacity()
        
        # 4. Data features demonstration
        demonstrate_data_features()
        
        print("\n" + "="*60)
        print("✅ DATA GENERATION DEMONSTRATION COMPLETE")
        print("="*60)
        print("📁 Check the 'simulation_data' directory for generated files")
        print("🚀 You can now scale up to generate larger datasets!")
        
        # Show generated files
        data_dir = Path("simulation_data")
        if data_dir.exists():
            files = list(data_dir.glob("*.csv"))
            print(f"\n📄 Generated files ({len(files)}):")
            for f in sorted(files)[-5:]:  # Show last 5 files
                size_mb = f.stat().st_size / (1024*1024)
                print(f"   • {f.name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
        raise

if __name__ == "__main__":
    main() 