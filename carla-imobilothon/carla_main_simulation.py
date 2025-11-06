"""
CARLA Simulation for i.mobilothon 5.0 - Volkswagen Group
Complete simulation covering all scenarios from architecture diagram
"""

import carla
import random
import time
import numpy as np
import cv2
import os
from datetime import datetime
import json
import math

class IMobilothonSimulation:
    def __init__(self):
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.sensors = {}
        self.data_collected = {
            'images': [],
            'imu_data': [],
            'gnss_data': [],
            'detected_objects': []
        }
        self.output_dir = "carla_simulation_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def connect_to_carla(self, host='localhost', port=2000, timeout=10.0):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA server at {host}:{port}...")
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print("✓ Successfully connected to CARLA!")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            print("\nMake sure CARLA server is running!")
            print("Run: CarlaUE4.exe (Windows) or ./CarlaUE4.sh (Linux)")
            return False
    
    def setup_world_scenario(self, scenario_type='highway'):
        """
        Setup different world scenarios
        scenario_type: 'highway', 'urban_traffic', 'foggy', 'night'
        """
        print(f"\nSetting up scenario: {scenario_type}")
        
        # Available maps in CARLA
        maps = {
            'highway': 'Town04',  # Highway with multiple lanes
            'urban': 'Town03',    # Urban area with traffic
            'complex': 'Town05',  # Complex urban scenario
            'rural': 'Town07'     # Rural roads
        }
        
        # Load appropriate map
        if scenario_type in ['highway', 'clean_highway']:
            map_name = maps['highway']
        elif scenario_type in ['traffic', 'urban_traffic']:
            map_name = maps['urban']
        else:
            map_name = maps['highway']
        
        print(f"Loading map: {map_name}")
        self.world = self.client.load_world(map_name)
        time.sleep(2)  # Wait for world to load
        
        # Get world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enable synchronous mode
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Setup weather based on scenario
        self.setup_weather(scenario_type)
        
        print(f"✓ Scenario '{scenario_type}' setup complete!")
    
    def setup_weather(self, scenario_type):
        """Setup weather conditions"""
        weather = carla.WeatherParameters()
        
        if scenario_type == 'foggy' or 'hazy' in scenario_type:
            # Foggy/Hazy conditions
            weather.cloudiness = 80.0
            weather.fog_density = 50.0
            weather.fog_distance = 20.0
            weather.wetness = 30.0
            print("  Weather: Foggy/Hazy conditions")
        elif scenario_type == 'night':
            # Night conditions
            weather.sun_altitude_angle = -90.0
            weather.cloudiness = 20.0
            print("  Weather: Night time")
        elif scenario_type == 'rain':
            # Rainy conditions
            weather.precipitation = 80.0
            weather.precipitation_deposits = 50.0
            weather.wetness = 80.0
            weather.cloudiness = 90.0
            print("  Weather: Rainy conditions")
        else:
            # Clear weather
            weather.cloudiness = 10.0
            weather.sun_altitude_angle = 70.0
            print("  Weather: Clear conditions")
        
        self.world.set_weather(weather)
    
    def spawn_ego_vehicle(self, vehicle_type='vehicle.tesla.model3'):
        """Spawn the ego vehicle (our car with sensors)"""
        print("\nSpawning ego vehicle...")
        
        # Get vehicle blueprint
        bp = self.blueprint_library.find(vehicle_type)
        
        # Get random spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        
        # Set autopilot for automatic driving
        self.vehicle.set_autopilot(True)
        
        print(f"✓ Ego vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def spawn_traffic_vehicles(self, num_vehicles=30):
        """Spawn traffic vehicles for traffic scenario"""
        print(f"\nSpawning {num_vehicles} traffic vehicles...")
        
        vehicles_list = []
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        # Get all vehicle blueprints
        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        
        for i in range(min(num_vehicles, len(spawn_points))):
            try:
                bp = random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(bp, spawn_points[i])
                vehicle.set_autopilot(True)
                vehicles_list.append(vehicle)
            except:
                continue
        
        print(f"✓ Spawned {len(vehicles_list)} traffic vehicles")
        return vehicles_list
    
    def spawn_pedestrians(self, num_pedestrians=20):
        """Spawn pedestrians"""
        print(f"\nSpawning {num_pedestrians} pedestrians...")
        
        pedestrians_list = []
        spawn_points = []
        
        # Get random spawn locations
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # Spawn pedestrians
        ped_bps = self.blueprint_library.filter('walker.pedestrian.*')
        
        for spawn_point in spawn_points:
            try:
                bp = random.choice(ped_bps)
                pedestrian = self.world.spawn_actor(bp, spawn_point)
                pedestrians_list.append(pedestrian)
            except:
                continue
        
        print(f"✓ Spawned {len(pedestrians_list)} pedestrians")
        return pedestrians_list
    
    def spawn_obstacles(self):
        """Spawn static obstacles (debris, stalled vehicles, etc.)"""
        print("\nSpawning obstacles...")
        
        obstacles = []
        
        # Get spawn location ahead of ego vehicle
        vehicle_location = self.vehicle.get_location()
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        
        # Spawn stalled vehicle on road
        stalled_bp = self.blueprint_library.find('vehicle.volkswagen.t2')
        stalled_transform = carla.Transform(
            carla.Location(
                x=vehicle_location.x + vehicle_forward.x * 50,
                y=vehicle_location.y + vehicle_forward.y * 50,
                z=vehicle_location.z + 0.5
            ),
            carla.Rotation(yaw=random.uniform(0, 360))
        )
        
        try:
            stalled_vehicle = self.world.spawn_actor(stalled_bp, stalled_transform)
            obstacles.append({'type': 'stalled_vehicle', 'actor': stalled_vehicle})
            print("  ✓ Spawned stalled vehicle")
        except:
            print("  ✗ Could not spawn stalled vehicle")
        
        # Spawn debris (using static objects)
        debris_bps = [
            'static.prop.container',
            'static.prop.barrel',
            'static.prop.box03'
        ]
        
        for i, debris_bp_name in enumerate(debris_bps):
            try:
                debris_bp = self.blueprint_library.find(debris_bp_name)
                debris_transform = carla.Transform(
                    carla.Location(
                        x=vehicle_location.x + vehicle_forward.x * (30 + i*10),
                        y=vehicle_location.y + vehicle_forward.y * (30 + i*10) + random.uniform(-2, 2),
                        z=vehicle_location.z + 0.5
                    )
                )
                debris = self.world.spawn_actor(debris_bp, debris_transform)
                obstacles.append({'type': 'debris', 'actor': debris})
                print(f"  ✓ Spawned debris {i+1}")
            except:
                continue
        
        print(f"✓ Total obstacles spawned: {len(obstacles)}")
        return obstacles
    
    def attach_cameras(self):
        """Attach multiple cameras with different configurations"""
        print("\nAttaching cameras...")
        
        # Camera configurations
        camera_configs = {
            'front_rgb': {
                'type': 'sensor.camera.rgb',
                'transform': carla.Transform(carla.Location(x=1.5, z=2.4)),
                'attributes': {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }
            },
            'front_fisheye': {
                'type': 'sensor.camera.rgb',
                'transform': carla.Transform(carla.Location(x=1.5, z=2.4)),
                'attributes': {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '170',  # Fisheye effect
                    'lens_circle_multiplier': '3.0',
                    'lens_circle_falloff': '3.0'
                }
            },
            'left_camera': {
                'type': 'sensor.camera.rgb',
                'transform': carla.Transform(
                    carla.Location(x=1.0, y=-0.5, z=2.0),
                    carla.Rotation(yaw=-45)
                ),
                'attributes': {
                    'image_size_x': '1280',
                    'image_size_y': '720',
                    'fov': '90'
                }
            },
            'right_camera': {
                'type': 'sensor.camera.rgb',
                'transform': carla.Transform(
                    carla.Location(x=1.0, y=0.5, z=2.0),
                    carla.Rotation(yaw=45)
                ),
                'attributes': {
                    'image_size_x': '1280',
                    'image_size_y': '720',
                    'fov': '90'
                }
            },
            'rear_camera': {
                'type': 'sensor.camera.rgb',
                'transform': carla.Transform(
                    carla.Location(x=-2.0, z=2.0),
                    carla.Rotation(yaw=180)
                ),
                'attributes': {
                    'image_size_x': '1280',
                    'image_size_y': '720',
                    'fov': '90'
                }
            }
        }
        
        for cam_name, config in camera_configs.items():
            bp = self.blueprint_library.find(config['type'])
            
            # Set attributes
            for attr_name, attr_value in config['attributes'].items():
                bp.set_attribute(attr_name, attr_value)
            
            # Spawn and attach camera
            camera = self.world.spawn_actor(
                bp,
                config['transform'],
                attach_to=self.vehicle
            )
            
            # Setup callback
            camera.listen(lambda image, name=cam_name: self.process_image(image, name))
            
            self.sensors[cam_name] = camera
            print(f"  ✓ Attached {cam_name}")
        
        print(f"✓ Total cameras attached: {len(self.sensors)}")
    
    def attach_imu_sensor(self):
        """Attach IMU sensor (Acceleration, Magnitude)"""
        print("\nAttaching IMU sensor...")
        
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=0))
        
        imu = self.world.spawn_actor(
            imu_bp,
            imu_transform,
            attach_to=self.vehicle
        )
        
        imu.listen(lambda data: self.process_imu(data))
        self.sensors['imu'] = imu
        
        print("✓ IMU sensor attached")
    
    def attach_gnss_sensor(self):
        """Attach GNSS sensor (GPS)"""
        print("\nAttaching GNSS sensor...")
        
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0, z=0))
        
        gnss = self.world.spawn_actor(
            gnss_bp,
            gnss_transform,
            attach_to=self.vehicle
        )
        
        gnss.listen(lambda data: self.process_gnss(data))
        self.sensors['gnss'] = gnss
        
        print("✓ GNSS sensor attached")
    
    def process_image(self, image, camera_name):
        """Process camera images with Adaptive Keyframe Sampling logic"""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Calculate image quality metrics (for AKS)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        
        # Check for motion (simplified)
        is_keyframe = self.is_keyframe(array, camera_name)
        
        if is_keyframe:
            # Save image
            timestamp = int(time.time() * 1000)
            filename = f"{self.output_dir}/{camera_name}_{timestamp}.png"
            cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            
            # Store metadata
            self.data_collected['images'].append({
                'camera': camera_name,
                'timestamp': timestamp,
                'filename': filename,
                'resolution': f"{image.width}x{image.height}"
            })
    
    def is_keyframe(self, current_frame, camera_name):
        """
        Adaptive Keyframe Sampling (AKS) logic
        Detects: video stream, motion, scene content, significant changes
        """
        # For simplicity, sample every Nth frame based on activity
        # In production, implement motion detection, scene change detection
        
        frame_count = len([img for img in self.data_collected['images'] 
                          if img['camera'] == camera_name])
        
        # Sample rate: every 10th frame for static scenes, every frame for dynamic
        if frame_count % 10 == 0:
            return True
        
        return False
    
    def process_imu(self, imu_data):
        """Process IMU sensor data"""
        acceleration = imu_data.accelerometer
        gyroscope = imu_data.gyroscope
        
        # Calculate magnitude
        accel_magnitude = math.sqrt(
            acceleration.x**2 + 
            acceleration.y**2 + 
            acceleration.z**2
        )
        
        data = {
            'timestamp': imu_data.timestamp,
            'acceleration': {
                'x': acceleration.x,
                'y': acceleration.y,
                'z': acceleration.z,
                'magnitude': accel_magnitude
            },
            'gyroscope': {
                'x': gyroscope.x,
                'y': gyroscope.y,
                'z': gyroscope.z
            }
        }
        
        self.data_collected['imu_data'].append(data)
    
    def process_gnss(self, gnss_data):
        """Process GNSS sensor data"""
        data = {
            'timestamp': gnss_data.timestamp,
            'latitude': gnss_data.latitude,
            'longitude': gnss_data.longitude,
            'altitude': gnss_data.altitude
        }
        
        self.data_collected['gnss_data'].append(data)
    
    def detect_objects_in_scene(self):
        """
        Detect objects in scene (stalled vehicles, debris, potholes, etc.)
        This simulates what your detection models would do
        """
        # Get all actors in the scene
        actors = self.world.get_actors()
        vehicle_location = self.vehicle.get_location()
        
        detected = []
        
        for actor in actors:
            # Skip ego vehicle
            if actor.id == self.vehicle.id:
                continue
            
            actor_location = actor.get_location()
            distance = vehicle_location.distance(actor_location)
            
            # Only detect objects within 50 meters
            if distance < 50:
                obj_type = 'unknown'
                
                if 'vehicle' in actor.type_id:
                    # Check if vehicle is moving
                    velocity = actor.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    if speed < 0.1:  # Nearly stationary
                        obj_type = 'stalled_vehicle'
                    else:
                        obj_type = 'moving_vehicle'
                
                elif 'walker' in actor.type_id:
                    obj_type = 'pedestrian'
                
                elif 'static' in actor.type_id or 'prop' in actor.type_id:
                    obj_type = 'debris'
                
                detected.append({
                    'type': obj_type,
                    'distance': distance,
                    'location': {
                        'x': actor_location.x,
                        'y': actor_location.y,
                        'z': actor_location.z
                    }
                })
        
        return detected
    
    def run_scenario(self, scenario_name, duration=60):
        """
        Run a complete scenario
        scenario_name: Name of the scenario
        duration: Duration in seconds
        """
        print(f"\n{'='*60}")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Tick the world
                self.world.tick()
                
                # Detect objects every 10 frames
                if frame_count % 10 == 0:
                    detected = self.detect_objects_in_scene()
                    if detected:
                        self.data_collected['detected_objects'].extend(detected)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {elapsed:.1f}s / {duration}s - Frames: {frame_count}")
        
        except KeyboardInterrupt:
            print("\n  Scenario interrupted by user")
        
        print(f"\n✓ Scenario '{scenario_name}' completed!")
        print(f"  Duration: {time.time() - start_time:.1f}s")
        print(f"  Frames processed: {frame_count}")
        self.print_statistics()
    
    def print_statistics(self):
        """Print collection statistics"""
        print("\n--- Data Collection Statistics ---")
        print(f"  Images collected: {len(self.data_collected['images'])}")
        print(f"  IMU readings: {len(self.data_collected['imu_data'])}")
        print(f"  GNSS readings: {len(self.data_collected['gnss_data'])}")
        print(f"  Objects detected: {len(self.data_collected['detected_objects'])}")
    
    def save_metadata(self):
        """Save all collected metadata to JSON"""
        metadata_file = f"{self.output_dir}/metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(self.data_collected, f, indent=2)
        
        print(f"\n✓ Metadata saved to {metadata_file}")
    
    def cleanup(self):
        """Cleanup all spawned actors and sensors"""
        print("\nCleaning up...")
        
        # Destroy sensors
        for sensor in self.sensors.values():
            if sensor.is_alive:
                sensor.destroy()
        
        # Destroy ego vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        print("✓ Cleanup complete")


def main():
    """Main function to run all scenarios"""
    print("="*60)
    print("CARLA SIMULATION FOR i.mobilothon 5.0")
    print("Volkswagen Group - Architecture Diagram Implementation")
    print("="*60)
    
    sim = IMobilothonSimulation()
    
    # Connect to CARLA
    if not sim.connect_to_carla():
        return
    
    # List of scenarios to run
    scenarios = [
        {
            'name': 'Clean Highway - Clear Weather',
            'world_type': 'clean_highway',
            'traffic': 5,
            'duration': 30
        },
        {
            'name': 'Highway with Traffic',
            'world_type': 'highway',
            'traffic': 30,
            'duration': 30
        },
        {
            'name': 'Foggy/Hazy Conditions',
            'world_type': 'foggy',
            'traffic': 15,
            'duration': 30
        },
        {
            'name': 'Obstacles Detection',
            'world_type': 'highway',
            'traffic': 10,
            'obstacles': True,
            'duration': 30
        }
    ]
    
    try:
        for scenario in scenarios:
            print(f"\n\n{'#'*60}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"{'#'*60}\n")
            
            # Setup world
            sim.setup_world_scenario(scenario['world_type'])
            
            # Spawn ego vehicle
            sim.spawn_ego_vehicle()
            
            # Attach sensors
            sim.attach_cameras()
            sim.attach_imu_sensor()
            sim.attach_gnss_sensor()
            
            # Spawn traffic
            if scenario.get('traffic', 0) > 0:
                sim.spawn_traffic_vehicles(scenario['traffic'])
            
            # Spawn obstacles if needed
            if scenario.get('obstacles', False):
                sim.spawn_obstacles()
            
            # Run scenario
            sim.run_scenario(scenario['name'], scenario['duration'])
            
            # Cleanup for next scenario
            sim.cleanup()
            time.sleep(2)
        
        # Save all metadata
        sim.save_metadata()
        
        print("\n" + "="*60)
        print("ALL SCENARIOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutput saved to: {sim.output_dir}/")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()