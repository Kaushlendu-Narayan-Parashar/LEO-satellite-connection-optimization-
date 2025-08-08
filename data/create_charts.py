#!/usr/bin/env python3
"""
LEO Satellite Data Visualization Script

This script reads CSV data from the LEO simulation and creates beautiful,
informative charts for analysis and presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class LEOChartGenerator:
    """Generate comprehensive charts from LEO satellite simulation data."""
    
    def __init__(self, csv_file_path):
        """Initialize with CSV data file."""
        self.csv_path = Path(csv_file_path)
        self.data = None
        self.output_dir = Path("charts")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📊 LEO Satellite Data Visualization")
        print(f"📁 Input file: {self.csv_path}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_data(self):
        """Load and prepare data for visualization."""
        print(f"\n📖 Loading data from {self.csv_path}...")
        
        try:
            # Load CSV with timestamp as index
            self.data = pd.read_csv(self.csv_path, index_col='timestamp_utc', parse_dates=True)
            
            print(f"✅ Loaded {len(self.data):,} data points")
            print(f"   • Columns: {len(self.data.columns)}")
            print(f"   • Time span: {(self.data.index.max() - self.data.index.min()).total_seconds() / 3600:.1f} hours")
            
            if 'satellite_name' in self.data.columns:
                unique_sats = self.data['satellite_name'].nunique()
                print(f"   • Satellites: {unique_sats}")
                
            # Add derived columns if not present
            if 'hour_utc' not in self.data.columns:
                self.data['hour_utc'] = self.data.index.hour
                
            # Ensure categorical columns exist
            if 'elevation_range' not in self.data.columns and 'elevation_deg' in self.data.columns:
                self.data['elevation_range'] = pd.cut(self.data['elevation_deg'], 
                                                     bins=[0, 10, 30, 60, 90], 
                                                     labels=['Low (5-10°)', 'Medium (10-30°)', 'High (30-60°)', 'Very High (60-90°)'])
            
            if 'signal_quality' not in self.data.columns and 'cn0_dBHz' in self.data.columns:
                self.data['signal_quality'] = pd.cut(self.data['cn0_dBHz'], 
                                                    bins=[-np.inf, 40, 60, 80, np.inf],
                                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_overview_dashboard(self):
        """Create a comprehensive overview dashboard."""
        print(f"\n📊 Creating overview dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LEO Satellite Communication Link Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Elevation vs C/N0 Scatter Plot
        if 'elevation_deg' in self.data.columns and 'cn0_dBHz' in self.data.columns:
            scatter_data = self.data.dropna(subset=['elevation_deg', 'cn0_dBHz'])
            scatter = axes[0, 0].scatter(scatter_data['elevation_deg'], scatter_data['cn0_dBHz'], 
                                       alpha=0.6, s=20, c=scatter_data.index.astype(np.int64), cmap='viridis')
            axes[0, 0].set_xlabel('Elevation Angle (degrees)')
            axes[0, 0].set_ylabel('C/N0 (dB-Hz)')
            axes[0, 0].set_title('Link Quality vs Elevation')
            plt.colorbar(scatter, ax=axes[0, 0], label='Time')
        
        # 2. Time Series of Key Parameters
        if len(self.data) > 0:
            # Plot elevation and C/N0 over time
            time_minutes = (self.data.index - self.data.index[0]).total_seconds() / 60
            
            ax1 = axes[0, 1]
            if 'elevation_deg' in self.data.columns:
                ax1.plot(time_minutes, self.data['elevation_deg'], 'b-', linewidth=2, label='Elevation')
                ax1.set_ylabel('Elevation (degrees)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            if 'cn0_dBHz' in self.data.columns:
                ax2.plot(time_minutes, self.data['cn0_dBHz'], 'r-', linewidth=2, label='C/N0')
                ax2.set_ylabel('C/N0 (dB-Hz)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.set_xlabel('Time (minutes)')
            ax1.set_title('Pass Timeline')
        
        # 3. Path Loss Breakdown
        loss_columns = [col for col in self.data.columns if 'loss' in col.lower() and col != 'total_path_loss_dB']
        if loss_columns and len(self.data) > 0:
            time_minutes = (self.data.index - self.data.index[0]).total_seconds() / 60
            for col in loss_columns[:4]:  # Show top 4 loss components
                if col in ['fspl_dB', 'total_atmospheric_loss_dB', 'rain_fade_dB', 'gaseous_loss_dB']:
                    axes[0, 2].plot(time_minutes, self.data[col], linewidth=2, label=col.replace('_', ' ').title())
            
            axes[0, 2].set_xlabel('Time (minutes)')
            axes[0, 2].set_ylabel('Path Loss (dB)')
            axes[0, 2].set_title('Path Loss Components')
            axes[0, 2].legend()
        
        # 4. MODCOD Distribution (if available)
        if 'selected_modcod' in self.data.columns:
            modcod_counts = self.data['selected_modcod'].value_counts().head(8)
            if len(modcod_counts) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(modcod_counts)))
                wedges, texts, autotexts = axes[1, 0].pie(modcod_counts.values, labels=modcod_counts.index, 
                                                         autopct='%1.1f%%', colors=colors, startangle=90)
                axes[1, 0].set_title('MODCOD Distribution')
                # Make text smaller for better fit
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
        
        # 5. Signal Quality by Satellite Type (if available)
        if 'satellite_type' in self.data.columns and 'cn0_dBHz' in self.data.columns:
            self.data.boxplot(column='cn0_dBHz', by='satellite_type', ax=axes[1, 1])
            axes[1, 1].set_title('Signal Quality by Satellite Type')
            axes[1, 1].set_xlabel('Satellite Type')
            axes[1, 1].set_ylabel('C/N0 (dB-Hz)')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        elif 'satellite_name' in self.data.columns and 'cn0_dBHz' in self.data.columns:
            # Fallback to satellite names if types not available
            unique_sats = self.data['satellite_name'].unique()[:5]  # Show top 5
            sat_data = self.data[self.data['satellite_name'].isin(unique_sats)]
            sat_data.boxplot(column='cn0_dBHz', by='satellite_name', ax=axes[1, 1])
            axes[1, 1].set_title('Signal Quality by Satellite')
            axes[1, 1].set_xlabel('Satellite')
            axes[1, 1].set_ylabel('C/N0 (dB-Hz)')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Data Rate vs Link Margin (if available)
        if 'data_rate_mbps' in self.data.columns and 'link_margin_dB' in self.data.columns:
            good_data = self.data[(self.data['data_rate_mbps'] > 0) & (self.data['link_margin_dB'] > -50)]
            if len(good_data) > 0:
                axes[1, 2].scatter(good_data['link_margin_dB'], good_data['data_rate_mbps'], alpha=0.6, s=30)
                axes[1, 2].set_xlabel('Link Margin (dB)')
                axes[1, 2].set_ylabel('Data Rate (Mbps)')
                axes[1, 2].set_title('Data Rate vs Link Margin')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / 'overview_dashboard.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"✅ Overview dashboard saved: {dashboard_path}")
        
        return fig
    
    def create_satellite_comparison_chart(self):
        """Create satellite-by-satellite comparison charts."""
        if 'satellite_name' not in self.data.columns:
            print("⚠️  No satellite names found, skipping satellite comparison")
            return None
            
        print(f"\n🛰️  Creating satellite comparison charts...")
        
        unique_satellites = self.data['satellite_name'].unique()
        n_sats = len(unique_satellites)
        
        if n_sats == 0:
            return None
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Satellite-by-Satellite Comparison', fontsize=16, fontweight='bold')
        
        # 1. Max elevation by satellite
        if 'elevation_deg' in self.data.columns:
            max_elevations = self.data.groupby('satellite_name')['elevation_deg'].max().sort_values(ascending=False)
            max_elevations.plot(kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('Maximum Elevation by Satellite')
            axes[0, 0].set_ylabel('Max Elevation (degrees)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average C/N0 by satellite
        if 'cn0_dBHz' in self.data.columns:
            avg_cn0 = self.data.groupby('satellite_name')['cn0_dBHz'].mean().sort_values(ascending=False)
            avg_cn0.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
            axes[0, 1].set_title('Average C/N0 by Satellite')
            axes[0, 1].set_ylabel('Average C/N0 (dB-Hz)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Pass duration by satellite (number of data points as proxy)
        pass_durations = self.data.groupby('satellite_name').size().sort_values(ascending=False)
        pass_durations.plot(kind='bar', ax=axes[1, 0], color='orange')
        axes[1, 0].set_title('Total Observation Time by Satellite')
        axes[1, 0].set_ylabel('Number of Data Points')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Link availability by satellite
        if 'selected_modcod' in self.data.columns:
            link_availability = self.data.groupby('satellite_name').apply(
                lambda x: (x['selected_modcod'] != 'Link Down').mean() * 100
            ).sort_values(ascending=False)
            link_availability.plot(kind='bar', ax=axes[1, 1], color='coral')
            axes[1, 1].set_title('Link Availability by Satellite')
            axes[1, 1].set_ylabel('Link Up Percentage (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save comparison chart
        comparison_path = self.output_dir / 'satellite_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✅ Satellite comparison chart saved: {comparison_path}")
        
        return fig
    
    def create_performance_analysis_charts(self):
        """Create detailed performance analysis charts."""
        print(f"\n📈 Creating performance analysis charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Link Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. C/N0 Distribution Histogram
        if 'cn0_dBHz' in self.data.columns:
            cn0_data = self.data['cn0_dBHz'].dropna()
            axes[0, 0].hist(cn0_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(cn0_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cn0_data.mean():.1f} dB-Hz')
            axes[0, 0].axvline(cn0_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {cn0_data.median():.1f} dB-Hz')
            axes[0, 0].set_xlabel('C/N0 (dB-Hz)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('C/N0 Distribution')
            axes[0, 0].legend()
        
        # 2. Elevation vs Path Loss
        if 'elevation_deg' in self.data.columns and 'total_path_loss_dB' in self.data.columns:
            scatter_data = self.data.dropna(subset=['elevation_deg', 'total_path_loss_dB'])
            axes[0, 1].scatter(scatter_data['elevation_deg'], scatter_data['total_path_loss_dB'], alpha=0.6, s=20)
            axes[0, 1].set_xlabel('Elevation Angle (degrees)')
            axes[0, 1].set_ylabel('Total Path Loss (dB)')
            axes[0, 1].set_title('Path Loss vs Elevation')
            
            # Add trend line
            if len(scatter_data) > 1:
                z = np.polyfit(scatter_data['elevation_deg'], scatter_data['total_path_loss_dB'], 1)
                p = np.poly1d(z)
                axes[0, 1].plot(scatter_data['elevation_deg'].sort_values(), 
                               p(scatter_data['elevation_deg'].sort_values()), 
                               "r--", alpha=0.8, linewidth=2, label='Trend')
                axes[0, 1].legend()
        
        # 3. Signal Quality Categories
        if 'signal_quality' in self.data.columns:
            quality_counts = self.data['signal_quality'].value_counts()
            colors = ['red', 'orange', 'lightgreen', 'green'][:len(quality_counts)]
            quality_counts.plot(kind='bar', ax=axes[1, 0], color=colors)
            axes[1, 0].set_title('Signal Quality Distribution')
            axes[1, 0].set_ylabel('Number of Observations')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Data Rate Analysis (if available)
        if 'data_rate_mbps' in self.data.columns:
            data_rate_data = self.data[self.data['data_rate_mbps'] > 0]['data_rate_mbps']
            if len(data_rate_data) > 0:
                axes[1, 1].hist(data_rate_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1, 1].axvline(data_rate_data.mean(), color='blue', linestyle='--', linewidth=2, 
                                  label=f'Mean: {data_rate_data.mean():.1f} Mbps')
                axes[1, 1].set_xlabel('Data Rate (Mbps)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Achievable Data Rate Distribution')
                axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save performance analysis
        performance_path = self.output_dir / 'performance_analysis.png'
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance analysis charts saved: {performance_path}")
        
        return fig
    
    def create_time_series_analysis(self):
        """Create detailed time series analysis."""
        print(f"\n⏰ Creating time series analysis...")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Convert index to minutes for better visualization
        time_minutes = (self.data.index - self.data.index[0]).total_seconds() / 60
        
        # 1. Multi-parameter time series
        ax1 = axes[0]
        if 'elevation_deg' in self.data.columns:
            ax1.plot(time_minutes, self.data['elevation_deg'], 'b-', linewidth=2, label='Elevation (deg)')
            ax1.set_ylabel('Elevation (degrees)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
        
        ax1_twin = ax1.twinx()
        if 'cn0_dBHz' in self.data.columns:
            ax1_twin.plot(time_minutes, self.data['cn0_dBHz'], 'r-', linewidth=2, label='C/N0 (dB-Hz)')
            ax1_twin.set_ylabel('C/N0 (dB-Hz)', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Elevation and Signal Quality Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Path loss components
        if 'total_path_loss_dB' in self.data.columns:
            axes[1].plot(time_minutes, self.data['total_path_loss_dB'], 'k-', linewidth=2, label='Total Path Loss')
            
            if 'fspl_dB' in self.data.columns:
                axes[1].plot(time_minutes, self.data['fspl_dB'], '--', linewidth=2, label='Free Space Path Loss')
            if 'total_atmospheric_loss_dB' in self.data.columns:
                axes[1].plot(time_minutes, self.data['total_atmospheric_loss_dB'], ':', linewidth=2, label='Atmospheric Loss')
            
            axes[1].set_ylabel('Path Loss (dB)')
            axes[1].set_title('Path Loss Components Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Data rate over time (if available)
        if 'data_rate_mbps' in self.data.columns:
            axes[2].plot(time_minutes, self.data['data_rate_mbps'], 'g-', linewidth=2, label='Data Rate')
            axes[2].fill_between(time_minutes, self.data['data_rate_mbps'], alpha=0.3, color='green')
            axes[2].set_ylabel('Data Rate (Mbps)')
            axes[2].set_xlabel('Time (minutes)')
            axes[2].set_title('Achievable Data Rate Over Time')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save time series analysis
        timeseries_path = self.output_dir / 'time_series_analysis.png'
        plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
        print(f"✅ Time series analysis saved: {timeseries_path}")
        
        return fig
    
    def create_all_charts(self):
        """Create all chart types."""
        print(f"\n🎨 Creating all visualization charts...")
        
        charts_created = []
        
        # Create each chart type
        try:
            fig1 = self.create_overview_dashboard()
            if fig1:
                charts_created.append("Overview Dashboard")
            
            fig2 = self.create_satellite_comparison_chart()
            if fig2:
                charts_created.append("Satellite Comparison")
            
            fig3 = self.create_performance_analysis_charts()
            if fig3:
                charts_created.append("Performance Analysis")
            
            fig4 = self.create_time_series_analysis()
            if fig4:
                charts_created.append("Time Series Analysis")
            
            plt.close('all')  # Close all figures to free memory
            
        except Exception as e:
            print(f"❌ Error creating charts: {e}")
            return False
        
        # Summary
        print(f"\n✅ CHART GENERATION COMPLETE!")
        print(f"📁 Charts saved in: {self.output_dir}/")
        print(f"🎯 Charts created: {len(charts_created)}")
        for chart in charts_created:
            print(f"   • {chart}")
        
        # List all files created
        chart_files = list(self.output_dir.glob("*.png"))
        if chart_files:
            print(f"\n📄 Chart files:")
            for file in sorted(chart_files):
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.1f} KB)")
        
        return True

def main():
    """Main function for chart generation."""
    parser = argparse.ArgumentParser(description='Generate charts from LEO satellite simulation data')
    parser.add_argument('csv_file', nargs='?', default=None, 
                       help='Path to CSV data file (if not provided, will search for latest)')
    parser.add_argument('--output-dir', default='charts', 
                       help='Output directory for charts (default: charts)')
    
    args = parser.parse_args()
    
    # Find CSV file if not provided
    csv_file = args.csv_file
    if not csv_file:
        data_dir = Path("simulation_data")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                # Use the most recent CSV file
                csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 Using latest CSV file: {csv_file}")
            else:
                print("❌ No CSV files found in simulation_data directory")
                print("💡 Run 'python3 generate_chart_data.py' first to generate data")
                return
        else:
            print("❌ No simulation_data directory found")
            print("💡 Run 'python3 generate_chart_data.py' first to generate data")
            return
    
    # Create chart generator
    generator = LEOChartGenerator(csv_file)
    generator.output_dir = Path(args.output_dir)
    generator.output_dir.mkdir(exist_ok=True)
    
    # Load data and create charts
    if generator.load_data():
        generator.create_all_charts()
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main() 