#!/usr/bin/env python3
"""
Massive LEO Satellite Data Generation Script

This script demonstrates how to generate very large datasets for AI training.
It shows different scales and provides estimates for production use.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from leo_simulation import SimulationDataGenerator
from leo_simulation.data_generator import SimulationConfig

def generate_large_scale_data(num_satellites=50, passes_per_satellite=3):
    """Generate a large-scale dataset."""
    print(f"\n🚀 GENERATING LARGE-SCALE DATASET")
    print(f"   Target: {num_satellites} satellites × {passes_per_satellite} passes = {num_satellites * passes_per_satellite} total passes")
    print("="*80)
    
    start_time = time.time()
    
    # Create generator
    generator = SimulationDataGenerator()
    
    # Load satellite data
    print("📡 Loading satellite catalog...")
    satellites = generator.load_satellite_data()
    print(f"✅ Available satellites: {len(satellites):,}")
    
    # Select diverse satellite types
    satellite_types = [
        # ISS and space stations
        'ISS (ZARYA)',
        
        # Weather satellites
        'NOAA 18', 'NOAA 19', 'NOAA 20', 'NOAA 21',
        'METOP-A', 'METOP-B', 'METOP-C',
        
        # Earth observation
        'TERRA', 'AQUA', 'AURA',
        'LANDSAT 8', 'LANDSAT 9',
        'SENTINEL-1A', 'SENTINEL-1B',
        'SENTINEL-2A', 'SENTINEL-2B',
        'SENTINEL-3A', 'SENTINEL-3B',
        'SENTINEL-5P',
        
        # Scientific satellites
        'GOES-16', 'GOES-17', 'GOES-18',
        'HIMAWARI-8', 'HIMAWARI-9',
        
        # Communication satellites
        'INTELSAT 901', 'INTELSAT 902', 'INTELSAT 903',
        'EUTELSAT 5 WEST A', 'EUTELSAT 7A', 'EUTELSAT 9B',
        
        # Navigation
        'GPS IIF-1', 'GPS IIF-2', 'GPS IIF-3',
        'GALILEO-101', 'GALILEO-102', 'GALILEO-103',
        
        # Research satellites
        'CALIPSO', 'CLOUDSAT', 'GRACE-1', 'GRACE-2',
        'ICESat-2', 'SMAP', 'SWOT',
        
        # CubeSats and small satellites
        'DOVE PIONEER', 'FLOCK 4P-1', 'FLOCK 4P-2',
    ]
    
    # Find available satellites from our target list
    available_targets = [sat for sat in satellite_types if sat in satellites]
    
    # If we need more satellites, add from the general catalog
    if len(available_targets) < num_satellites:
        all_sat_names = list(satellites.keys())
        # Prefer satellites with common names (likely more active)
        additional_sats = [sat for sat in all_sat_names 
                          if sat not in available_targets 
                          and any(keyword in sat.upper() for keyword in 
                                ['NOAA', 'LANDSAT', 'TERRA', 'AQUA', 'GOES', 'METOP', 'SENTINEL'])]
        available_targets.extend(additional_sats[:num_satellites - len(available_targets)])
    
    # Final selection
    selected_satellites = available_targets[:num_satellites]
    
    print(f"🛰️  Selected {len(selected_satellites)} satellites:")
    for i, sat in enumerate(selected_satellites[:10], 1):
        print(f"   {i:2d}. {sat}")
    if len(selected_satellites) > 10:
        print(f"   ... and {len(selected_satellites) - 10} more")
    
    # Generate the dataset
    print(f"\n⚡ Starting data generation...")
    generation_start = time.time()
    
    dataset = generator.generate_multi_satellite_dataset(
        satellite_names=selected_satellites,
        max_passes_per_satellite=passes_per_satellite
    )
    
    generation_time = time.time() - generation_start
    
    if dataset.empty:
        print("❌ No data generated")
        return None
    
    # Save the dataset
    print(f"\n💾 Saving dataset...")
    save_start = time.time()
    saved_files = generator.save_dataset(dataset, filename_prefix=f"large_scale_{num_satellites}sats")
    save_time = time.time() - save_start
    
    # Generate comprehensive analysis
    print(f"📊 Generating analysis report...")
    analysis_start = time.time()
    report = generator.generate_analysis_report(dataset)
    analysis_time = time.time() - analysis_start
    
    total_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n" + "="*80)
    print(f"📈 LARGE-SCALE GENERATION RESULTS")
    print(f"="*80)
    
    # Basic statistics
    unique_satellites = dataset['satellite_name'].nunique() if 'satellite_name' in dataset.columns else 0
    unique_passes = dataset['pass_number'].nunique() if 'pass_number' in dataset.columns else 0
    time_span_hours = (dataset.index.max() - dataset.index.min()).total_seconds() / 3600 if len(dataset) > 0 else 0
    
    print(f"📊 DATASET STATISTICS:")
    print(f"   • Total data points: {len(dataset):,}")
    print(f"   • Unique satellites: {unique_satellites}")
    print(f"   • Total passes: {unique_passes}")
    print(f"   • Columns per record: {len(dataset.columns)}")
    print(f"   • Time span: {time_span_hours:.1f} hours")
    print(f"   • Memory usage: {dataset.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    
    print(f"\n⏱️  PERFORMANCE METRICS:")
    print(f"   • Total generation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   • Data generation: {generation_time:.1f}s ({generation_time/total_time*100:.1f}%)")
    print(f"   • File saving: {save_time:.1f}s ({save_time/total_time*100:.1f}%)")
    print(f"   • Analysis: {analysis_time:.1f}s ({analysis_time/total_time*100:.1f}%)")
    print(f"   • Processing rate: {len(dataset)/generation_time:.0f} points/second")
    print(f"   • Pass generation rate: {unique_passes/generation_time:.1f} passes/second")
    
    # File information
    for format_type, file_path in saved_files.items():
        if format_type == 'csv':
            file_size_mb = Path(file_path).stat().st_size / (1024*1024)
            print(f"   • {format_type.upper()} file size: {file_size_mb:.1f} MB")
    
    print(f"\n🎯 DATA QUALITY METRICS:")
    if 'cn0_dBHz' in dataset.columns:
        cn0_stats = dataset['cn0_dBHz'].describe()
        print(f"   • C/N0 range: {cn0_stats['min']:.1f} to {cn0_stats['max']:.1f} dB-Hz")
        print(f"   • C/N0 mean: {cn0_stats['mean']:.1f} dB-Hz (std: {cn0_stats['std']:.1f})")
    
    if 'elevation_deg' in dataset.columns:
        elev_stats = dataset['elevation_deg'].describe()
        print(f"   • Elevation range: {elev_stats['min']:.1f}° to {elev_stats['max']:.1f}°")
        print(f"   • Mean elevation: {elev_stats['mean']:.1f}°")
    
    if 'selected_modcod' in dataset.columns:
        link_up_pct = (dataset['selected_modcod'] != 'Link Down').mean() * 100
        print(f"   • Link availability: {link_up_pct:.1f}%")
        
        if link_up_pct > 0:
            good_data = dataset[dataset['selected_modcod'] != 'Link Down']
            avg_data_rate = good_data['data_rate_mbps'].mean()
            print(f"   • Average data rate: {avg_data_rate:.1f} Mbps")
    
    # Per-satellite breakdown
    if 'satellite_name' in dataset.columns and unique_satellites <= 20:
        print(f"\n📡 PER-SATELLITE BREAKDOWN:")
        sat_summary = dataset.groupby('satellite_name').agg({
            'elevation_deg': ['count', 'max'],
            'cn0_dBHz': 'mean'
        }).round(1)
        
        for sat_name in sat_summary.index[:10]:  # Show top 10
            count = int(sat_summary.loc[sat_name, ('elevation_deg', 'count')])
            max_elev = sat_summary.loc[sat_name, ('elevation_deg', 'max')]
            avg_cn0 = sat_summary.loc[sat_name, ('cn0_dBHz', 'mean')]
            print(f"   • {sat_name}: {count} points, max elev: {max_elev}°, avg C/N0: {avg_cn0:.1f} dB-Hz")
        
        if len(sat_summary) > 10:
            print(f"   ... and {len(sat_summary) - 10} more satellites")
    
    print(f"\n💾 OUTPUT FILES:")
    for format_type, file_path in saved_files.items():
        print(f"   • {format_type}: {file_path}")
    
    return dataset

def estimate_production_capacity():
    """Estimate capacity for production-scale data generation."""
    print(f"\n🏭 PRODUCTION CAPACITY ESTIMATION")
    print("="*80)
    
    # Run a benchmark
    print("🔍 Running benchmark...")
    benchmark_start = time.time()
    
    generator = SimulationDataGenerator()
    satellites = generator.load_satellite_data()
    
    # Test with a few satellites
    test_satellites = ['ISS (ZARYA)', 'NOAA 18', 'NOAA 19']
    available_test_sats = [sat for sat in test_satellites if sat in satellites]
    
    if not available_test_sats:
        print("❌ No test satellites available")
        return
    
    benchmark_data = generator.generate_multi_satellite_dataset(
        satellite_names=available_test_sats[:2],  # Use 2 satellites
        max_passes_per_satellite=1
    )
    
    benchmark_time = time.time() - benchmark_start
    
    if benchmark_data.empty:
        print("❌ Benchmark failed")
        return
    
    # Calculate performance metrics
    points_per_second = len(benchmark_data) / benchmark_time
    passes_generated = benchmark_data['pass_number'].nunique() if 'pass_number' in benchmark_data.columns else len(available_test_sats)
    passes_per_second = passes_generated / benchmark_time
    
    print(f"📊 BENCHMARK RESULTS:")
    print(f"   • Test data points: {len(benchmark_data)}")
    print(f"   • Test passes: {passes_generated}")
    print(f"   • Benchmark time: {benchmark_time:.1f} seconds")
    print(f"   • Processing rate: {points_per_second:.0f} points/second")
    print(f"   • Pass rate: {passes_per_second:.2f} passes/second")
    
    # Extrapolate to production scales
    print(f"\n🚀 PRODUCTION SCALE ESTIMATES:")
    
    production_scenarios = [
        ("Research Dataset", 100, 2, "Small research project"),
        ("Training Dataset", 500, 5, "Medium AI training"),
        ("Production Dataset", 1000, 10, "Large-scale AI training"),
        ("Enterprise Dataset", 2000, 20, "Enterprise AI system"),
        ("Massive Dataset", 5000, 50, "Comprehensive analysis")
    ]
    
    for scenario_name, num_sats, passes_per_sat, description in production_scenarios:
        total_passes = num_sats * passes_per_sat
        estimated_points = total_passes * (len(benchmark_data) / passes_generated)  # Average points per pass
        estimated_time_hours = total_passes / passes_per_second / 3600
        estimated_storage_mb = estimated_points * (benchmark_data.memory_usage(deep=True).sum() / len(benchmark_data)) / (1024*1024)
        estimated_storage_gb = estimated_storage_mb / 1024
        
        print(f"\n   📋 {scenario_name}:")
        print(f"      • Scale: {num_sats} satellites × {passes_per_sat} passes = {total_passes:,} passes")
        print(f"      • Estimated data points: ~{estimated_points:,.0f}")
        print(f"      • Estimated generation time: ~{estimated_time_hours:.1f} hours")
        print(f"      • Estimated storage: ~{estimated_storage_gb:.1f} GB")
        print(f"      • Use case: {description}")
        
        # Cost estimates (rough)
        if estimated_time_hours < 1:
            time_desc = f"{estimated_time_hours*60:.0f} minutes"
        elif estimated_time_hours < 24:
            time_desc = f"{estimated_time_hours:.1f} hours"
        else:
            time_desc = f"{estimated_time_hours/24:.1f} days"
        
        print(f"      • Practical time: {time_desc}")
    
    # Hardware recommendations
    print(f"\n💻 HARDWARE RECOMMENDATIONS:")
    print(f"   • CPU: Multi-core processor (4+ cores recommended)")
    print(f"   • RAM: 8GB+ (16GB+ for large datasets)")
    print(f"   • Storage: SSD recommended for faster I/O")
    print(f"   • Network: Stable internet for TLE data downloads")
    
    # Optimization tips
    print(f"\n⚡ OPTIMIZATION TIPS:")
    print(f"   • Cache TLE data to avoid repeated downloads")
    print(f"   • Use parallel processing for multiple satellites")
    print(f"   • Generate data in batches for very large datasets")
    print(f"   • Use Parquet format for better compression")
    print(f"   • Consider cloud computing for massive datasets")

def main():
    """Main function for massive data generation demonstration."""
    print("🛰️  MASSIVE LEO SATELLITE DATA GENERATION")
    print("="*80)
    print("This script demonstrates large-scale data generation capabilities")
    
    try:
        # Ask user for scale
        print("\nSelect generation scale:")
        print("1. Medium scale (20 satellites × 2 passes = 40 passes)")
        print("2. Large scale (50 satellites × 3 passes = 150 passes)")
        print("3. Production scale (100 satellites × 5 passes = 500 passes)")
        print("4. Just show capacity estimates")
        
        choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            dataset = generate_large_scale_data(num_satellites=20, passes_per_satellite=2)
        elif choice == "2":
            dataset = generate_large_scale_data(num_satellites=50, passes_per_satellite=3)
        elif choice == "3":
            dataset = generate_large_scale_data(num_satellites=100, passes_per_satellite=5)
        elif choice == "4":
            estimate_production_capacity()
            return
        else:
            print("Invalid choice, using default (medium scale)")
            dataset = generate_large_scale_data(num_satellites=20, passes_per_satellite=2)
        
        # Show capacity estimates
        estimate_production_capacity()
        
        print(f"\n" + "="*80)
        print(f"✅ MASSIVE DATA GENERATION COMPLETE")
        print(f"="*80)
        print(f"📁 Check the 'simulation_data' directory for your generated dataset")
        print(f"🚀 Scale up further by modifying the parameters in this script!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error in massive data generation: {e}")
        raise

if __name__ == "__main__":
    main() 