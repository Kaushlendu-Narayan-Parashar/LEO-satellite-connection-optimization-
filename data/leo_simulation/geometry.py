"""
Part I: The Dynamic Geometry of LEO Links

This module implements satellite orbital mechanics and geometry calculations
using the skyfield library for SGP4 propagation and coordinate transformations.
"""

import pandas as pd
import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield.timelib import Time
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SatelliteGeometry:
    """
    Handles satellite orbital mechanics and geometry calculations.
    
    This class manages TLE data loading, satellite tracking, and calculation
    of time-varying link geometry parameters (elevation, azimuth, slant range).
    """
    
    def __init__(self):
        """Initialize the geometry calculator with skyfield components."""
        logger.info("Initializing SatelliteGeometry...")
        
        # Initialize Skyfield loader and timescale
        self.loader = load
        self.eph = load('de421.bsp')  # Planetary ephemeris
        self.earth = self.eph['earth']
        self.ts = load.timescale()
        
        # Storage for loaded satellites and ground stations
        self.satellites = {}
        self.ground_stations = {}
        
        logger.info("Skyfield timescale and ephemeris loaded successfully")
    
    def load_tle_data(self, tle_url: str = None, filename: str = 'active.tle', 
                      max_age_days: float = 1.0) -> Dict[str, Any]:
        """
        Load TLE data from CelesTrak or local file.
        
        Args:
            tle_url: URL to download TLE data from
            filename: Local filename to cache TLE data
            max_age_days: Maximum age of cached file before re-downloading
            
        Returns:
            Dictionary mapping satellite names to EarthSatellite objects
        """
        if tle_url is None:
            tle_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
        
        logger.info(f"Loading TLE data from {filename}...")
        
        # Download if file doesn't exist or is too old
        if not self.loader.exists(filename) or self.loader.days_old(filename) > max_age_days:
            logger.info(f"Downloading fresh TLE data to '{filename}'...")
            self.loader.download(tle_url, filename=filename)
        else:
            logger.info(f"Using cached TLE data '{filename}' (less than {max_age_days} days old)")
        
        # Parse TLE file
        satellites = self.loader.tle_file(filename)
        self.satellites = {sat.name: sat for sat in satellites}
        
        logger.info(f"Loaded {len(self.satellites)} satellites")
        return self.satellites
    
    def add_ground_station(self, name: str, lat_deg: float, lon_deg: float, 
                          alt_m: float = 0.0) -> None:
        """
        Add a ground station location.
        
        Args:
            name: Ground station identifier
            lat_deg: Latitude in degrees (positive North)
            lon_deg: Longitude in degrees (positive East)  
            alt_m: Altitude in meters above WGS84 ellipsoid
        """
        ground_station = wgs84.latlon(
            latitude_degrees=lat_deg,
            longitude_degrees=lon_deg,
            elevation_m=alt_m
        )
        
        self.ground_stations[name] = {
            'object': ground_station,
            'lat': lat_deg,
            'lon': lon_deg, 
            'alt_m': alt_m
        }
        
        logger.info(f"Added ground station '{name}' at {lat_deg:.2f}°N, {lon_deg:.2f}°E, {alt_m}m")
    
    def find_satellite_passes(self, satellite_name: str, ground_station_name: str,
                             start_time: Time, end_time: Time, 
                             min_elevation_deg: float = 5.0) -> pd.DataFrame:
        """
        Find all visible passes of a satellite over a ground station.
        
        Args:
            satellite_name: Name of satellite in loaded TLE data
            ground_station_name: Name of ground station
            start_time: Start of search window
            end_time: End of search window
            min_elevation_deg: Minimum elevation for visibility
            
        Returns:
            DataFrame with pass events (rise, culminate, set times)
        """
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite '{satellite_name}' not found in loaded TLE data")
        
        if ground_station_name not in self.ground_stations:
            raise ValueError(f"Ground station '{ground_station_name}' not defined")
        
        satellite = self.satellites[satellite_name]
        ground_station = self.ground_stations[ground_station_name]['object']
        
        logger.info(f"Finding passes for {satellite_name} over {ground_station_name}...")
        
        # Find rise, culmination, and set events
        times, events = satellite.find_events(
            ground_station, start_time, end_time, 
            altitude_degrees=min_elevation_deg
        )
        
        event_names = ['rise', 'culminate', 'set']
        
        # Create DataFrame with pass events
        pass_events = []
        for i, (time, event) in enumerate(zip(times, events)):
            pass_events.append({
                'time_utc': time.utc_datetime(),
                'event': event_names[event],
                'event_code': event
            })
        
        passes_df = pd.DataFrame(pass_events)
        logger.info(f"Found {len(passes_df)} pass events")
        
        return passes_df
    
    def generate_pass_geometry(self, satellite_name: str, ground_station_name: str,
                              rise_time: Time, set_time: Time, 
                              time_step_seconds: float = 10.0) -> pd.DataFrame:
        """
        Generate detailed geometry data for a complete satellite pass.
        
        Args:
            satellite_name: Name of satellite
            ground_station_name: Name of ground station
            rise_time: Pass start time
            set_time: Pass end time
            time_step_seconds: Time resolution in seconds
            
        Returns:
            DataFrame with time-series geometry data
        """
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite '{satellite_name}' not found")
        
        if ground_station_name not in self.ground_stations:
            raise ValueError(f"Ground station '{ground_station_name}' not found")
        
        satellite = self.satellites[satellite_name]
        ground_station = self.ground_stations[ground_station_name]['object']
        
        logger.info(f"Generating geometry for {satellite_name} pass over {ground_station_name}")
        logger.info(f"Pass duration: {rise_time.utc_strftime('%H:%M:%S')} to {set_time.utc_strftime('%H:%M:%S')} UTC")
        
        # Generate time array for the pass
        pass_duration_days = set_time - rise_time
        num_steps = int(pass_duration_days * 24 * 3600 / time_step_seconds)
        time_range = self.ts.linspace(rise_time, set_time, num_steps)
        
        # Perform vectorized geometry calculation
        difference = satellite - ground_station
        topocentric = difference.at(time_range)
        alt, az, dist = topocentric.altaz()
        
        # Create DataFrame with results
        geometry_df = pd.DataFrame({
            'timestamp_utc': time_range.utc_datetime(),
            'elevation_deg': alt.degrees,
            'azimuth_deg': az.degrees,
            'slant_range_km': dist.km,
            'satellite': satellite_name,
            'ground_station': ground_station_name
        })
        
        geometry_df.set_index('timestamp_utc', inplace=True)
        
        logger.info(f"Generated geometry data for {len(geometry_df)} time steps")
        return geometry_df
    
    def get_next_pass_geometry(self, satellite_name: str, ground_station_name: str,
                              start_search_time: Time = None, 
                              time_step_seconds: float = 10.0,
                              min_elevation_deg: float = 5.0) -> Optional[pd.DataFrame]:
        """
        Find and generate geometry for the next visible satellite pass.
        
        Args:
            satellite_name: Name of satellite
            ground_station_name: Name of ground station  
            start_search_time: When to start looking for passes (default: now)
            time_step_seconds: Time resolution for geometry calculation
            min_elevation_deg: Minimum elevation for visibility
            
        Returns:
            DataFrame with pass geometry, or None if no pass found
        """
        if start_search_time is None:
            start_search_time = self.ts.now()
        
        # Search for passes in next 24 hours
        end_search_time = self.ts.utc(
            start_search_time.utc.year,
            start_search_time.utc.month, 
            start_search_time.utc.day + 1
        )
        
        # Find pass events
        passes_df = self.find_satellite_passes(
            satellite_name, ground_station_name,
            start_search_time, end_search_time,
            min_elevation_deg
        )
        
        if len(passes_df) == 0:
            logger.warning("No passes found in search window")
            return None
        
        # Find first complete pass (rise -> culminate -> set)
        rise_events = passes_df[passes_df['event'] == 'rise']
        if len(rise_events) == 0:
            logger.warning("No rise events found")
            return None
        
        # Get the first rise time
        first_rise_idx = rise_events.index[0]
        rise_time = self.ts.utc(rise_events.iloc[0]['time_utc'])
        
        # Find corresponding set time
        set_events = passes_df[(passes_df.index > first_rise_idx) & 
                              (passes_df['event'] == 'set')]
        if len(set_events) == 0:
            logger.warning("No corresponding set event found")
            return None
        
        set_time = self.ts.utc(set_events.iloc[0]['time_utc'])
        
        # Generate detailed geometry for this pass
        return self.generate_pass_geometry(
            satellite_name, ground_station_name,
            rise_time, set_time, time_step_seconds
        )


def create_default_delhi_station() -> Tuple[SatelliteGeometry, str]:
    """
    Convenience function to create a geometry calculator with Delhi ground station.
    
    Returns:
        Tuple of (SatelliteGeometry instance, ground station name)
    """
    geo = SatelliteGeometry()
    
    # Add Delhi, India as default ground station
    geo.add_ground_station(
        name='Delhi',
        lat_deg=28.61,
        lon_deg=77.23, 
        alt_m=216.0
    )
    
    return geo, 'Delhi'


# Make the file directly executable
if __name__ == "__main__":
    """Run geometry calculation when executed directly."""
    print("LEO Satellite Geometry Calculator")
    print("=" * 40)
    
    try:
        # Initialize geometry calculator
        print("Initializing satellite geometry calculator...")
        geo = SatelliteGeometry()
        
        # Add Delhi ground station
        geo.add_ground_station(
            name='Delhi',
            lat_deg=28.61,
            lon_deg=77.23, 
            alt_m=216.0
        )
        print("✓ Ground station added: Delhi")
        
        # Load satellite TLE data
        print("\nLoading satellite TLE data...")
        satellites = geo.load_tle_data()
        print(f"✓ Loaded {len(satellites)} satellites")
        
        # List some available satellites
        sat_names = list(satellites.keys())[:10]
        print(f"\nFirst 10 available satellites:")
        for i, name in enumerate(sat_names, 1):
            print(f"  {i:2d}. {name}")
        
        # Generate geometry for ISS pass
        print(f"\nSearching for ISS pass...")
        satellite_name = 'ISS (ZARYA)'
        
        if satellite_name not in satellites:
            print(f"❌ {satellite_name} not found in TLE data")
            # Try alternative ISS names
            iss_alternatives = [name for name in satellites.keys() if 'ISS' in name]
            if iss_alternatives:
                satellite_name = iss_alternatives[0]
                print(f"Using alternative: {satellite_name}")
            else:
                print("No ISS found, using first available satellite")
                satellite_name = sat_names[0]
        
        # Get next pass geometry
        geometry_df = geo.get_next_pass_geometry(
            satellite_name=satellite_name,
            ground_station_name='Delhi',
            time_step_seconds=10.0,
            min_elevation_deg=5.0
        )
        
        if geometry_df is None:
            print("❌ No passes found in the next 24 hours")
        else:
            print(f"✓ Generated geometry data for {len(geometry_df)} time steps")
            
            # Display basic statistics
            print(f"\n📊 Pass Statistics:")
            print(f"  Duration: {len(geometry_df) * 10 / 60:.1f} minutes")
            print(f"  Max elevation: {geometry_df['elevation_deg'].max():.1f}°")
            print(f"  Min elevation: {geometry_df['elevation_deg'].min():.1f}°")
            print(f"  Slant range: {geometry_df['slant_range_km'].min():.1f} - {geometry_df['slant_range_km'].max():.1f} km")
            
            # Show sample data
            print(f"\n📋 Sample Geometry Data:")
            print(geometry_df.head(10).round(2))
            
            # Save data to CSV
            output_file = 'satellite_geometry_data.csv'
            geometry_df.to_csv(output_file)
            print(f"\n💾 Data saved to: {output_file}")
            
            print(f"\n✅ Geometry calculation completed successfully!")
            print(f"📄 Output file: {output_file} ({len(geometry_df)} rows)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 