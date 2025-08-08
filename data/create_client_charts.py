#!/usr/bin/env python3
"""
Client-Focused LEO Satellite Chart Generator

This script creates charts specifically optimized for client presentations
with multiple export formats and professional styling.
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

# Client-focused styling
plt.style.use('default')
sns.set_palette("Set2")  # Professional, colorblind-friendly palette
plt.rcParams['figure.figsize'] = (16, 10)  # Large, presentation-ready size
plt.rcParams['font.size'] = 12  # Larger font for readability
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ClientChartGenerator:
    """Generate client-focused charts with multiple export formats."""
    
    def __init__(self, csv_file_path):
        """Initialize with CSV data file."""
        self.csv_path = Path(csv_file_path)
        self.data = None
        self.output_dir = Path("client_charts")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 CLIENT-FOCUSED CHART GENERATOR")
        print(f"📁 Input: {self.csv_path}")
        print(f"📁 Output: {self.output_dir}")
        
    def load_data(self):
        """Load and prepare data for client visualization."""
        try:
            self.data = pd.read_csv(self.csv_path, index_col='timestamp_utc', parse_dates=True)
            print(f"✅ Loaded {len(self.data):,} data points")
            
            # Add derived columns for client-friendly analysis
            if 'elevation_range' not in self.data.columns and 'elevation_deg' in self.data.columns:
                self.data['elevation_range'] = pd.cut(self.data['elevation_deg'], 
                                                     bins=[0, 15, 45, 75, 90], 
                                                     labels=['Low', 'Medium', 'High', 'Excellent'])
            
            if 'performance_category' not in self.data.columns and 'cn0_dBHz' in self.data.columns:
                self.data['performance_category'] = pd.cut(self.data['cn0_dBHz'], 
                                                          bins=[-np.inf, 30, 50, 70, np.inf],
                                                          labels=['Poor', 'Fair', 'Good', 'Excellent'])
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_executive_summary_chart(self):
        """Create a single-page executive summary chart."""
        print(f"📊 Creating executive summary chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('LEO Satellite Communication Performance - Executive Summary', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Key Performance Indicator (top-left)
        if 'cn0_dBHz' in self.data.columns:
            cn0_data = self.data['cn0_dBHz'].dropna()
            
            # Create KPI-style visualization
            axes[0, 0].text(0.5, 0.8, f"{cn0_data.mean():.1f}", 
                           transform=axes[0, 0].transAxes, fontsize=48, 
                           ha='center', va='center', fontweight='bold', color='darkblue')
            axes[0, 0].text(0.5, 0.6, "Average Signal Quality", 
                           transform=axes[0, 0].transAxes, fontsize=16, 
                           ha='center', va='center')
            axes[0, 0].text(0.5, 0.5, "(C/N0 in dB-Hz)", 
                           transform=axes[0, 0].transAxes, fontsize=12, 
                           ha='center', va='center', style='italic')
            
            # Add performance indicator
            if cn0_data.mean() > 60:
                performance = "EXCELLENT"
                color = 'green'
            elif cn0_data.mean() > 45:
                performance = "GOOD"
                color = 'orange'
            else:
                performance = "NEEDS IMPROVEMENT"
                color = 'red'
            
            axes[0, 0].text(0.5, 0.3, performance, 
                           transform=axes[0, 0].transAxes, fontsize=16, 
                           ha='center', va='center', fontweight='bold', color=color)
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].axis('off')
        
        # 2. Performance by Satellite (top-right)
        if 'satellite_name' in self.data.columns and 'cn0_dBHz' in self.data.columns:
            sat_performance = self.data.groupby('satellite_name')['cn0_dBHz'].mean().sort_values(ascending=True)
            colors = ['lightcoral' if x < 45 else 'lightgreen' if x > 60 else 'gold' for x in sat_performance.values]
            
            sat_performance.plot(kind='barh', ax=axes[0, 1], color=colors)
            axes[0, 1].set_title('Signal Quality by Satellite', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Average C/N0 (dB-Hz)')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. System Reliability (bottom-left)
        if 'selected_modcod' in self.data.columns:
            link_up = (self.data['selected_modcod'] != 'Link Down').sum()
            total_points = len(self.data)
            reliability_pct = (link_up / total_points) * 100
            
            # Create pie chart for reliability
            sizes = [reliability_pct, 100 - reliability_pct]
            labels = [f'Link Available\n{reliability_pct:.1f}%', f'Link Down\n{100-reliability_pct:.1f}%']
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=colors, 
                                                     autopct='', startangle=90, textprops={'fontsize': 12})
            axes[1, 0].set_title('System Reliability', fontsize=14, fontweight='bold')
        
        # 4. Performance Trends (bottom-right)
        if 'elevation_deg' in self.data.columns and 'cn0_dBHz' in self.data.columns:
            # Scatter plot with trend line
            scatter_data = self.data.dropna(subset=['elevation_deg', 'cn0_dBHz'])
            axes[1, 1].scatter(scatter_data['elevation_deg'], scatter_data['cn0_dBHz'], 
                             alpha=0.6, s=30, color='steelblue')
            
            # Add trend line
            if len(scatter_data) > 1:
                z = np.polyfit(scatter_data['elevation_deg'], scatter_data['cn0_dBHz'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(scatter_data['elevation_deg'].min(), 
                                    scatter_data['elevation_deg'].max(), 100)
                axes[1, 1].plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label='Trend')
            
            axes[1, 1].set_xlabel('Satellite Elevation Angle (degrees)')
            axes[1, 1].set_ylabel('Signal Quality (C/N0 dB-Hz)')
            axes[1, 1].set_title('Performance vs Elevation', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_path = self.output_dir / 'executive_summary'
        self.save_multiple_formats(fig, base_path, "Executive Summary")
        
        return fig
    
    def create_technical_deep_dive(self):
        """Create detailed technical analysis for technical stakeholders."""
        print(f"🔧 Creating technical deep dive...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('LEO Satellite Communication - Technical Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Signal Quality Distribution
        if 'cn0_dBHz' in self.data.columns:
            cn0_data = self.data['cn0_dBHz'].dropna()
            axes[0, 0].hist(cn0_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(cn0_data.mean(), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {cn0_data.mean():.1f} dB-Hz')
            axes[0, 0].axvline(cn0_data.median(), color='green', linestyle='--', linewidth=2, 
                              label=f'Median: {cn0_data.median():.1f} dB-Hz')
            axes[0, 0].set_xlabel('C/N0 (dB-Hz)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Signal Quality Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Path Loss Analysis
        if 'elevation_deg' in self.data.columns and 'total_path_loss_dB' in self.data.columns:
            scatter_data = self.data.dropna(subset=['elevation_deg', 'total_path_loss_dB'])
            axes[0, 1].scatter(scatter_data['elevation_deg'], scatter_data['total_path_loss_dB'], 
                             alpha=0.6, s=20, color='orange')
            axes[0, 1].set_xlabel('Elevation Angle (degrees)')
            axes[0, 1].set_ylabel('Total Path Loss (dB)')
            axes[0, 1].set_title('Path Loss vs Elevation')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss Component Breakdown
        loss_columns = ['fspl_dB', 'gaseous_loss_dB', 'rain_fade_dB', 'cloud_loss_dB']
        available_loss_cols = [col for col in loss_columns if col in self.data.columns]
        
        if available_loss_cols:
            loss_means = [self.data[col].mean() for col in available_loss_cols]
            col_labels = [col.replace('_dB', '').replace('_', ' ').title() for col in available_loss_cols]
            
            axes[0, 2].bar(col_labels, loss_means, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
            axes[0, 2].set_ylabel('Average Loss (dB)')
            axes[0, 2].set_title('Loss Component Breakdown')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. MODCOD Performance
        if 'selected_modcod' in self.data.columns:
            modcod_counts = self.data['selected_modcod'].value_counts().head(8)
            if len(modcod_counts) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(modcod_counts)))
                modcod_counts.plot(kind='bar', ax=axes[1, 0], color=colors)
                axes[1, 0].set_title('Modulation Scheme Usage')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Data Rate Analysis
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
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. System Performance Summary Table
        axes[1, 2].axis('off')
        
        # Create performance summary
        summary_data = []
        if 'cn0_dBHz' in self.data.columns:
            cn0_stats = self.data['cn0_dBHz'].describe()
            summary_data.extend([
                ['Average Signal Quality', f"{cn0_stats['mean']:.1f} dB-Hz"],
                ['Signal Quality Range', f"{cn0_stats['min']:.1f} to {cn0_stats['max']:.1f} dB-Hz"],
            ])
        
        if 'selected_modcod' in self.data.columns:
            reliability = (self.data['selected_modcod'] != 'Link Down').mean() * 100
            summary_data.append(['Link Reliability', f"{reliability:.1f}%"])
        
        if 'data_rate_mbps' in self.data.columns:
            avg_rate = self.data[self.data['data_rate_mbps'] > 0]['data_rate_mbps'].mean()
            summary_data.append(['Average Data Rate', f"{avg_rate:.1f} Mbps"])
        
        if 'satellite_name' in self.data.columns:
            summary_data.append(['Satellites Analyzed', f"{self.data['satellite_name'].nunique()}"])
        
        summary_data.append(['Total Observations', f"{len(self.data):,}"])
        
        # Create table
        table = axes[1, 2].table(cellText=summary_data, 
                                colLabels=['Metric', 'Value'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_path = self.output_dir / 'technical_analysis'
        self.save_multiple_formats(fig, base_path, "Technical Analysis")
        
        return fig
    
    def create_business_impact_chart(self):
        """Create business-focused impact visualization."""
        print(f"💼 Creating business impact chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('LEO Satellite Communication - Business Impact Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Service Availability (top-left)
        if 'selected_modcod' in self.data.columns:
            availability = (self.data['selected_modcod'] != 'Link Down').mean() * 100
            downtime = 100 - availability
            
            # Create gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Convert to Cartesian for plotting
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            axes[0, 0].plot(x, y, 'k-', linewidth=3)
            
            # Add availability arc
            avail_theta = np.linspace(0, np.pi * availability / 100, 50)
            avail_x = np.cos(avail_theta)
            avail_y = np.sin(avail_theta)
            axes[0, 0].plot(avail_x, avail_y, 'g-', linewidth=8, label=f'Available: {availability:.1f}%')
            
            # Add text
            axes[0, 0].text(0, -0.3, f"{availability:.1f}%", ha='center', va='center', 
                           fontsize=36, fontweight='bold', color='green')
            axes[0, 0].text(0, -0.5, "Service Availability", ha='center', va='center', fontsize=16)
            
            axes[0, 0].set_xlim(-1.2, 1.2)
            axes[0, 0].set_ylim(-0.6, 1.2)
            axes[0, 0].set_aspect('equal')
            axes[0, 0].axis('off')
            axes[0, 0].set_title('Service Availability', fontsize=14, fontweight='bold')
        
        # 2. Revenue Impact (top-right)
        if 'data_rate_mbps' in self.data.columns:
            data_rates = self.data[self.data['data_rate_mbps'] > 0]['data_rate_mbps']
            if len(data_rates) > 0:
                # Assume $1 per Mbps per hour as example revenue
                hourly_revenue = data_rates.mean()
                daily_revenue = hourly_revenue * 24
                monthly_revenue = daily_revenue * 30
                
                revenues = [hourly_revenue, daily_revenue, monthly_revenue]
                periods = ['Hourly', 'Daily', 'Monthly']
                colors = ['lightblue', 'lightgreen', 'gold']
                
                bars = axes[0, 1].bar(periods, revenues, color=colors)
                axes[0, 1].set_ylabel('Revenue Potential ($)')
                axes[0, 1].set_title('Revenue Impact Analysis\n(Example: $1/Mbps/hour)', fontsize=14, fontweight='bold')
                
                # Add value labels on bars
                for bar, revenue in zip(bars, revenues):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'${revenue:.0f}', ha='center', va='bottom', fontweight='bold')
                
                axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Performance Trends (bottom-left)
        if len(self.data) > 0:
            time_hours = (self.data.index - self.data.index[0]).total_seconds() / 3600
            
            if 'cn0_dBHz' in self.data.columns:
                # Smooth the data for trend visualization
                window = max(1, len(self.data) // 20)  # 5% of data points
                smoothed_cn0 = self.data['cn0_dBHz'].rolling(window=window, center=True).mean()
                
                axes[1, 0].plot(time_hours, smoothed_cn0, 'b-', linewidth=3, label='Signal Quality Trend')
                axes[1, 0].fill_between(time_hours, smoothed_cn0, alpha=0.3, color='blue')
                axes[1, 0].set_xlabel('Time (hours)')
                axes[1, 0].set_ylabel('Signal Quality (C/N0 dB-Hz)')
                axes[1, 0].set_title('Performance Trend Over Time', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
        
        # 4. Risk Assessment (bottom-right)
        risk_factors = []
        risk_levels = []
        
        if 'cn0_dBHz' in self.data.columns:
            avg_cn0 = self.data['cn0_dBHz'].mean()
            if avg_cn0 > 60:
                risk_factors.append('Signal Quality')
                risk_levels.append(1)  # Low risk
            elif avg_cn0 > 45:
                risk_factors.append('Signal Quality')
                risk_levels.append(2)  # Medium risk
            else:
                risk_factors.append('Signal Quality')
                risk_levels.append(3)  # High risk
        
        if 'selected_modcod' in self.data.columns:
            availability = (self.data['selected_modcod'] != 'Link Down').mean() * 100
            if availability > 95:
                risk_factors.append('Service Availability')
                risk_levels.append(1)
            elif availability > 90:
                risk_factors.append('Service Availability')
                risk_levels.append(2)
            else:
                risk_factors.append('Service Availability')
                risk_levels.append(3)
        
        if 'rain_fade_dB' in self.data.columns:
            avg_rain_fade = self.data['rain_fade_dB'].mean()
            if avg_rain_fade < 2:
                risk_factors.append('Weather Impact')
                risk_levels.append(1)
            elif avg_rain_fade < 5:
                risk_factors.append('Weather Impact')
                risk_levels.append(2)
            else:
                risk_factors.append('Weather Impact')
                risk_levels.append(3)
        
        if risk_factors:
            colors = ['green' if r == 1 else 'orange' if r == 2 else 'red' for r in risk_levels]
            bars = axes[1, 1].barh(risk_factors, risk_levels, color=colors)
            axes[1, 1].set_xlabel('Risk Level')
            axes[1, 1].set_xlim(0, 4)
            axes[1, 1].set_xticks([1, 2, 3])
            axes[1, 1].set_xticklabels(['Low', 'Medium', 'High'])
            axes[1, 1].set_title('Risk Assessment', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_path = self.output_dir / 'business_impact'
        self.save_multiple_formats(fig, base_path, "Business Impact")
        
        return fig
    
    def save_multiple_formats(self, fig, base_path, chart_name):
        """Save chart in multiple client-friendly formats."""
        formats = {
            'png': {'dpi': 300, 'format': 'png'},
            'pdf': {'dpi': 300, 'format': 'pdf'},
            'svg': {'format': 'svg'},
            'jpg': {'dpi': 300, 'format': 'jpg', 'facecolor': 'white'}
        }
        
        saved_files = []
        for format_name, kwargs in formats.items():
            try:
                file_path = base_path.with_suffix(f'.{format_name}')
                fig.savefig(file_path, bbox_inches='tight', **kwargs)
                file_size = file_path.stat().st_size / 1024  # KB
                saved_files.append(f"{format_name.upper()}: {file_path.name} ({file_size:.0f} KB)")
            except Exception as e:
                print(f"⚠️  Could not save {format_name}: {e}")
        
        print(f"✅ {chart_name} saved in {len(saved_files)} formats:")
        for file_info in saved_files:
            print(f"   • {file_info}")
    
    def create_all_client_charts(self):
        """Create all client-focused chart types."""
        print(f"\n🎨 Creating all client-focused charts...")
        
        charts_created = []
        
        try:
            # Executive Summary - for C-level executives
            fig1 = self.create_executive_summary_chart()
            if fig1:
                charts_created.append("Executive Summary")
            
            # Technical Deep Dive - for technical teams
            fig2 = self.create_technical_deep_dive()
            if fig2:
                charts_created.append("Technical Analysis")
            
            # Business Impact - for business stakeholders
            fig3 = self.create_business_impact_chart()
            if fig3:
                charts_created.append("Business Impact")
            
            plt.close('all')
            
        except Exception as e:
            print(f"❌ Error creating charts: {e}")
            return False
        
        print(f"\n✅ CLIENT CHART GENERATION COMPLETE!")
        print(f"📁 Charts saved in: {self.output_dir}/")
        print(f"🎯 Chart types created: {len(charts_created)}")
        for chart in charts_created:
            print(f"   • {chart}")
        
        # List all files created
        all_files = list(self.output_dir.glob("*.*"))
        if all_files:
            print(f"\n📄 All files created ({len(all_files)}):")
            for file in sorted(all_files):
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.0f} KB)")
        
        return True

def main():
    """Main function for client chart generation."""
    parser = argparse.ArgumentParser(description='Generate client-focused charts from LEO satellite data')
    parser.add_argument('csv_file', nargs='?', default=None, 
                       help='Path to CSV data file')
    
    args = parser.parse_args()
    
    # Find CSV file if not provided
    csv_file = args.csv_file
    if not csv_file:
        data_dir = Path("simulation_data")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 Using latest CSV file: {csv_file}")
            else:
                print("❌ No CSV files found. Generate data first.")
                return
        else:
            print("❌ No simulation_data directory found. Generate data first.")
            return
    
    # Create client chart generator
    generator = ClientChartGenerator(csv_file)
    
    # Load data and create charts
    if generator.load_data():
        generator.create_all_client_charts()
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main() 