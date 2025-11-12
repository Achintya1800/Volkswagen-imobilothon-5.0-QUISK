"""Geofencing utilities for hazard alerts"""
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates in meters
    
    Args:
        lat1, lon1: First coordinate
        lat2, lon2: Second coordinate
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    
    a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def is_within_geofence(current_lat, current_lon, hazard_lat, hazard_lon, radius_m=500):
    """
    Check if current location is within geofence radius of hazard
    
    Args:
        current_lat, current_lon: Current GPS coordinates
        hazard_lat, hazard_lon: Hazard GPS coordinates
        radius_m: Geofence radius in meters (default: 500m)
    
    Returns:
        bool: True if within geofence
    """
    distance = haversine_distance(current_lat, current_lon, hazard_lat, hazard_lon)
    return distance <= radius_m, distance
