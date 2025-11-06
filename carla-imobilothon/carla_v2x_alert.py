"""
V2X (Vehicle-to-Everything) Communication Simulation
Implements the Alert System from the architecture diagram:
- Connectivity Check
- Cloud Publish (MQTT/HTTPS to cloud)
- Local Cache (Store in device when offline)
- Alert Nearby Drivers via V2X
- Voice-based GENAI Verification
- In-cloud clustering to avoid deduplication
"""

import carla
import random
import time
import json
import math
from collections import defaultdict
from datetime import datetime


class V2XAlertSystem:
    def __init__(self, world):
        self.world = world
        self.vehicles = []
        self.alerts = []
        self.local_cache = []
        self.cloud_alerts = []
        self.is_online = True  # Simulate connectivity
        self.alert_history = defaultdict(list)  # For deduplication
        
    def add_vehicle(self, vehicle, vehicle_id):
        """Register a vehicle in the V2X network"""
        self.vehicles.append({
            'id': vehicle_id,
            'actor': vehicle,
            'alerts_received': [],
            'alerts_sent': []
        })
    
    def check_connectivity(self):
        """
        Simulate connectivity check
        Returns True if online, False if offline
        """
        # Simulate random connectivity issues (10% chance offline)
        if random.random() < 0.1:
            self.is_online = False
            print("  ⚠ Connectivity: OFFLINE - Storing alerts locally")
            return False
        else:
            self.is_online = True
            return True
    
    def create_alert(self, vehicle_id, hazard_type, location, severity='high'):
        """
        Create a hazard alert
        hazard_type: 'stalled_vehicle', 'debris', 'pothole', 'accident', etc.
        """
        alert = {
            'id': f"alert_{int(time.time() * 1000)}_{vehicle_id}",
            'timestamp': time.time(),
            'vehicle_id': vehicle_id,
            'hazard_type': hazard_type,
            'location': {
                'x': location.x,
                'y': location.y,
                'z': location.z
            },
            'severity': severity,
            'confidence': random.uniform(0.7, 0.99),
            'status': 'pending'
        }
        
        # Check if hazard already reported (deduplication)
        if not self.is_duplicate_alert(alert):
            self.alerts.append(alert)
            print(f"\n  🚨 NEW ALERT: {hazard_type} at ({location.x:.1f}, {location.y:.1f})")
            print(f"     Severity: {severity} | Confidence: {alert['confidence']:.2f}")
            
            # Process alert based on connectivity
            self.process_alert(alert)
            
            return alert
        else:
            print(f"  ℹ️  Duplicate alert filtered: {hazard_type}")
            return None
    
    def is_duplicate_alert(self, new_alert, distance_threshold=10.0, time_threshold=300):
        """
        Check if similar alert exists nearby (spatial + temporal clustering)
        distance_threshold: meters
        time_threshold: seconds
        """
        new_loc = new_alert['location']
        current_time = time.time()
        
        for existing_alert in self.alerts + self.cloud_alerts:
            # Check time difference
            time_diff = current_time - existing_alert['timestamp']
            if time_diff > time_threshold:
                continue
            
            # Check spatial distance
            exist_loc = existing_alert['location']
            distance = math.sqrt(
                (new_loc['x'] - exist_loc['x'])**2 +
                (new_loc['y'] - exist_loc['y'])**2
            )
            
            # Check hazard type
            if (distance < distance_threshold and 
                existing_alert['hazard_type'] == new_alert['hazard_type']):
                return True
        
        return False
    
    def process_alert(self, alert):
        """Process alert based on connectivity"""
        if self.is_online:
            # Push to cloud
            self.cloud_publish(alert)
            
            # Also check and upload local cache
            if self.local_cache:
                print(f"  ☁️  Uploading {len(self.local_cache)} cached alerts...")
                for cached_alert in self.local_cache:
                    self.cloud_publish(cached_alert)
                self.local_cache.clear()
        else:
            # Store in local cache
            self.store_local_cache(alert)
    
    def cloud_publish(self, alert):
        """
        Simulate publishing to cloud (MQTT/HTTPS)
        In production: Use AWS IoT, Azure IoT Hub, or custom MQTT broker
        """
        alert['status'] = 'published_to_cloud'
        alert['cloud_timestamp'] = time.time()
        
        self.cloud_alerts.append(alert)
        
        print(f"  ☁️  Published to cloud: {alert['id']}")
        
        # Simulate cloud processing and broadcast
        self.broadcast_to_nearby_vehicles(alert)
    
    def store_local_cache(self, alert):
        """Store alert in local device cache (offline mode)"""
        alert['status'] = 'cached_locally'
        self.local_cache.append(alert)
        print(f"  💾 Stored locally: {alert['id']} (will upload when online)")
    
    def broadcast_to_nearby_vehicles(self, alert, broadcast_radius=500.0):
        """
        Broadcast alert to nearby vehicles via V2X (DSRC/C-V2X)
        broadcast_radius: meters
        """
        alert_location = carla.Location(
            x=alert['location']['x'],
            y=alert['location']['y'],
            z=alert['location']['z']
        )
        
        alerted_count = 0
        
        for vehicle_data in self.vehicles:
            vehicle = vehicle_data['actor']
            vehicle_id = vehicle_data['id']
            
            # Skip the vehicle that sent the alert
            if vehicle_id == alert['vehicle_id']:
                continue
            
            vehicle_location = vehicle.get_location()
            distance = vehicle_location.distance(alert_location)
            
            # Check if within broadcast radius
            if distance < broadcast_radius:
                # Calculate haversine distance for geofencing
                haversine_dist = self.calculate_haversine_distance(
                    vehicle_location, alert_location
                )
                
                # Alert the vehicle
                self.alert_vehicle(vehicle_id, alert, distance)
                alerted_count += 1
        
        print(f"  📡 V2X Broadcast: Alerted {alerted_count} nearby vehicles")
    
    def calculate_haversine_distance(self, loc1, loc2):
        """
        Calculate haversine distance for geofencing
        Used in "Alert nearby drivers via V2X" with haversine distance calc
        """
        # In CARLA, coordinates are in meters, but for production:
        # Convert to lat/lon and use actual haversine formula
        
        # Simplified distance for CARLA coordinates
        return math.sqrt(
            (loc1.x - loc2.x)**2 + 
            (loc1.y - loc2.y)**2
        )
    
    def alert_vehicle(self, vehicle_id, alert, distance):
        """
        Send alert to specific vehicle
        Implements Voice-based GENAI verification
        """
        # Find vehicle data
        vehicle_data = next(
            (v for v in self.vehicles if v['id'] == vehicle_id), 
            None
        )
        
        if vehicle_data:
            vehicle_data['alerts_received'].append({
                'alert': alert,
                'distance': distance,
                'received_at': time.time()
            })
            
            # Voice-based GENAI verification simulation
            self.voice_genai_verification(vehicle_id, alert)
    
    def voice_genai_verification(self, vehicle_id, alert):
        """
        Voice-based GENAI verification from architecture diagram
        - Send to reputed drivers
        - Show blurred thumbnail
        - Trusted drivers voice-confirm
        - Voice response: Yes/No/Unsure
        - STT parses confirmation
        - Bayesian trust update
        """
        # Simulate trust score
        vehicle_trust_score = random.uniform(0.5, 1.0)
        
        if vehicle_trust_score > 0.7:  # Reputed driver
            print(f"  🎤 Voice verification request to vehicle {vehicle_id}")
            
            # Simulate voice response
            responses = ['Yes', 'No', 'Unsure']
            voice_response = random.choice(responses)
            
            # Parse confirmation (STT simulation)
            if voice_response == 'Yes':
                confirmation = 1.0
            elif voice_response == 'No':
                confirmation = 0.0
            else:
                confirmation = 0.5
            
            # Bayesian trust update
            updated_confidence = self.bayesian_trust_update(
                alert['confidence'],
                confirmation,
                vehicle_trust_score
            )
            
            alert['confidence'] = updated_confidence
            
            print(f"     Response: {voice_response} | " +
                  f"Updated confidence: {updated_confidence:.2f}")
    
    def bayesian_trust_update(self, prior_confidence, confirmation, trust_score):
        """
        Bayesian trust update formula
        Updates alert confidence based on driver feedback
        """
        # Simplified Bayesian update
        weight = trust_score  # Weight by driver's trust score
        
        # Weighted average
        updated = (prior_confidence + weight * confirmation) / (1 + weight)
        
        return min(max(updated, 0.0), 1.0)  # Clamp to [0, 1]
    
    def cluster_alerts_in_cloud(self):
        """
        In-cloud clustering to avoid deduplication
        Groups similar alerts by location and time
        """
        print("\n  🔄 Running cloud clustering...")
        
        clusters = []
        processed = set()
        
        for i, alert in enumerate(self.cloud_alerts):
            if i in processed:
                continue
            
            cluster = [alert]
            processed.add(i)
            
            # Find similar alerts
            for j, other_alert in enumerate(self.cloud_alerts[i+1:], start=i+1):
                if j in processed:
                    continue
                
                if self.alerts_are_similar(alert, other_alert):
                    cluster.append(other_alert)
                    processed.add(j)
            
            clusters.append(cluster)
        
        print(f"  ✓ Clustered {len(self.cloud_alerts)} alerts into {len(clusters)} groups")
        
        # Merge duplicates in each cluster
        merged_alerts = []
        for cluster in clusters:
            merged = self.merge_cluster(cluster)
            merged_alerts.append(merged)
        
        return merged_alerts
    
    def alerts_are_similar(self, alert1, alert2, 
                          distance_threshold=15.0, time_threshold=60):
        """Check if two alerts are similar enough to cluster"""
        # Check time
        time_diff = abs(alert1['timestamp'] - alert2['timestamp'])
        if time_diff > time_threshold:
            return False
        
        # Check location
        loc1 = alert1['location']
        loc2 = alert2['location']
        distance = math.sqrt(
            (loc1['x'] - loc2['x'])**2 + 
            (loc1['y'] - loc2['y'])**2
        )
        
        if distance > distance_threshold:
            return False
        
        # Check hazard type
        if alert1['hazard_type'] != alert2['hazard_type']:
            return False
        
        return True
    
    def merge_cluster(self, cluster):
        """Merge multiple alerts into one representative alert"""
        if len(cluster) == 1:
            return cluster[0]
        
        # Use weighted average of locations
        total_confidence = sum(a['confidence'] for a in cluster)
        
        merged_location = {
            'x': sum(a['location']['x'] * a['confidence'] for a in cluster) / total_confidence,
            'y': sum(a['location']['y'] * a['confidence'] for a in cluster) / total_confidence,
            'z': sum(a['location']['z'] * a['confidence'] for a in cluster) / total_confidence
        }
        
        # Take highest confidence
        max_confidence = max(a['confidence'] for a in cluster)
        
        merged = {
            'id': f"merged_{cluster[0]['id']}",
            'timestamp': min(a['timestamp'] for a in cluster),
            'vehicle_id': 'multiple',
            'hazard_type': cluster[0]['hazard_type'],
            'location': merged_location,
            'severity': cluster[0]['severity'],
            'confidence': max_confidence,
            'status': 'merged',
            'source_count': len(cluster)
        }
        
        return merged
    
    def get_statistics(self):
        """Get V2X system statistics"""
        stats = {
            'total_vehicles': len(self.vehicles),
            'total_alerts': len(self.alerts),
            'cloud_alerts': len(self.cloud_alerts),
            'cached_alerts': len(self.local_cache),
            'is_online': self.is_online
        }
        return stats
    
    def print_statistics(self):
        """Print V2X system statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("V2X ALERT SYSTEM STATISTICS")
        print("="*60)
        print(f"Total Vehicles in Network: {stats['total_vehicles']}")
        print(f"Alerts Created: {stats['total_alerts']}")
        print(f"Alerts in Cloud: {stats['cloud_alerts']}")
        print(f"Alerts in Local Cache: {stats['cached_alerts']}")
        print(f"Connectivity Status: {'ONLINE' if stats['is_online'] else 'OFFLINE'}")
        
        # Print alert breakdown
        hazard_counts = defaultdict(int)
        for alert in self.alerts:
            hazard_counts[alert['hazard_type']] += 1
        
        print("\nAlert Breakdown by Type:")
        for hazard_type, count in hazard_counts.items():
            print(f"  - {hazard_type}: {count}")
        
        print("="*60)


def run_v2x_simulation():
    """Main function to demonstrate V2X alert system"""
    print("="*60)
    print("V2X ALERT SYSTEM SIMULATION")
    print("="*60)
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Enable synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    try:
        # Create V2X system
        v2x = V2XAlertSystem(world)
        
        # Spawn multiple vehicles
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        
        vehicles = []
        for i in range(10):
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_points[i])
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)
                v2x.add_vehicle(vehicle, f"vehicle_{i}")
                print(f"✓ Spawned vehicle {i}")
            except:
                continue
        
        # Spawn some hazards
        print("\nSpawning hazards...")
        
        hazards = []
        hazard_types = ['stalled_vehicle', 'debris', 'debris']
        
        for i, hazard_type in enumerate(hazard_types):
            if hazard_type == 'stalled_vehicle':
                bp = blueprint_library.find('vehicle.volkswagen.t2')
            else:
                bp = blueprint_library.find('static.prop.barrel')
            
            spawn_point = spawn_points[len(vehicles) + i]
            hazard = world.spawn_actor(bp, spawn_point)
            hazards.append(hazard)
            print(f"✓ Spawned {hazard_type}")
        
        # Run simulation
        print("\nRunning V2X simulation...")
        
        for frame in range(600):  # 30 seconds
            world.tick()
            
            # Randomly check connectivity
            if frame % 50 == 0:
                v2x.check_connectivity()
            
            # Randomly create alerts when vehicles detect hazards
            if frame % 100 == 0:  # Every 5 seconds
                detecting_vehicle = random.choice(vehicles)
                vehicle_id = f"vehicle_{vehicles.index(detecting_vehicle)}"
                
                # Detect nearby hazard
                for hazard in hazards:
                    distance = detecting_vehicle.get_location().distance(
                        hazard.get_location()
                    )
                    
                    if distance < 30:  # Within 30 meters
                        hazard_type = 'stalled_vehicle' if 'vehicle' in hazard.type_id else 'debris'
                        
                        v2x.create_alert(
                            vehicle_id=vehicle_id,
                            hazard_type=hazard_type,
                            location=hazard.get_location(),
                            severity='high' if distance < 15 else 'medium'
                        )
                        break
            
            # Progress update
            if frame % 100 == 0:
                print(f"  Progress: {frame/20:.1f}s / 30s")
        
        # Run cloud clustering
        merged_alerts = v2x.cluster_alerts_in_cloud()
        
        # Print statistics
        v2x.print_statistics()
        
        print(f"\nAfter clustering: {len(merged_alerts)} unique hazards identified")
        
        # Cleanup
        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        for hazard in hazards:
            if hazard.is_alive:
                hazard.destroy()
        
        print("\n✓ V2X simulation completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    run_v2x_simulation()