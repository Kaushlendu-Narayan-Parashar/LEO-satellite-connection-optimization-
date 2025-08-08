#!/usr/bin/env python3
"""
One-Stop LEO Satellite Data Generation and Visualization

This script generates moderate amounts of satellite data and immediately
creates beautiful charts for analysis and presentation.
"""

import sys
import subprocess
from pathlib import Path
import time

def run_command(command, description):
    """Run a command and handle output."""
    print(f"\n🚀 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Main workflow function."""
    print("🛰️  LEO SATELLITE DATA GENERATION & VISUALIZATION WORKFLOW")
    print("="*70)
    print("This script will:")
    print("1. Generate moderate amount of satellite data (6 satellites, 3 passes each)")
    print("2. Create comprehensive charts and visualizations")
    print("3. Show you the results")
    
    start_time = time.time()
    
    # Step 1: Generate chart-optimized data
    print(f"\n📊 STEP 1: GENERATING CHART-OPTIMIZED DATA")
    print("="*50)
    
    success = run_command(
        "source leo_env/bin/activate && python3 generate_chart_data.py",
        "Generating satellite communication data"
    )
    
    if not success:
        print("❌ Data generation failed. Cannot proceed with charts.")
        return
    
    # Step 2: Create charts
    print(f"\n📈 STEP 2: CREATING VISUALIZATION CHARTS")
    print("="*50)
    
    success = run_command(
        "source leo_env/bin/activate && python3 create_charts.py",
        "Creating visualization charts"
    )
    
    if not success:
        print("⚠️  Chart generation had issues, but data is available.")
    
    # Step 3: Show results
    total_time = time.time() - start_time
    
    print(f"\n✅ WORKFLOW COMPLETE!")
    print("="*50)
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Check what was created
    data_dir = Path("simulation_data")
    chart_dir = Path("charts")
    
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"\n📊 DATA FILES CREATED ({len(csv_files)}):")
        for file in sorted(csv_files)[-3:]:  # Show last 3
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   • {file.name} ({size_mb:.1f} MB)")
    
    if chart_dir.exists():
        chart_files = list(chart_dir.glob("*.png"))
        print(f"\n📈 CHART FILES CREATED ({len(chart_files)}):")
        for file in sorted(chart_files):
            size_kb = file.stat().st_size / 1024
            print(f"   • {file.name} ({size_kb:.1f} KB)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. View charts in the 'charts/' directory")
    print(f"   2. Open CSV files in Excel or data analysis tools")
    print(f"   3. Use the data for your presentations or analysis")
    
    print(f"\n💡 QUICK VIEW COMMANDS:")
    if chart_dir.exists() and list(chart_dir.glob("*.png")):
        print(f"   • View charts: open charts/")
        print(f"   • Latest CSV: open simulation_data/")
    
    print(f"\n🚀 Your moderate dataset with beautiful charts is ready!")

if __name__ == "__main__":
    main() 