# server/clustering.py
from math import radians, sin, cos, sqrt, atan2

# Haversine: returns distance in meters
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000  # earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Find nearest hazard within threshold (meters); hazards list should have dicts with keys latitude, longitude, id
def find_nearby(hazards_list, lat, lon, threshold_meters=50):
    for h in hazards_list:
        d = haversine_meters(lat, lon, h['latitude'], h['longitude'])
        if d <= threshold_meters:
            return h['id']
    return None
