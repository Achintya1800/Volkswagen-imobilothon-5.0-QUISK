"""
Specialized scenario for detecting road defects:
- Potholes
- Unmarked speed bumps
- Road cracks
- Uneven pavement
"""

import carla
import random
import numpy as np
import cv2
import time

class RoadDefectScenario:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.blueprint_library = world.get_blueprint_library()
        self.defects = []
        
    def create_semantic_camera(self):
        """
        Create semantic segmentation camera to identify road surfaces
        This helps detect road vs pavement
        """
        bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', '1920')
        bp.set_attribute('image_size_y', '1080')
        bp.set_attribute('fov', '90')
        
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        camera.listen(lambda image: self.process_semantic_image(image))
        return camera
    
    def create_depth_camera(self):
        """
        Create depth camera to detect elevation changes
        Useful for detecting potholes and speed bumps
        """
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', '1920')
        bp.set_attribute('image_size_y', '1080')
        bp.set_attribute('fov', '90')
        
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        camera.listen(lambda image: self.process_depth_image(image))
        return camera
    
    def process_semantic_image(self, image):
        """
        Process semantic segmentation to identify:
        - Road surface (label 7)
        - Pavement (label 1)
        - Obstacles on road
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Extract red channel (contains class labels)
        semantic_labels = array[:, :, 2]
        
        # Identify road pixels (label 7)
        road_mask = (semantic_labels == 7)
        
        # Identify pavement/sidewalk (label 1)
        pavement_mask = (semantic_labels == 1)
        
        # Save for analysis
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f"carla_simulation_output/semantic_{timestamp}.png", semantic_labels)
        
    def process_depth_image(self, image):
        """
        Process depth image to detect elevation anomalies
        Sudden depth changes indicate potholes or speed bumps
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Convert to depth values (0-1 range, logarithmic)
        # Normalize for visualization
        depth_gray = array[:, :, 0]
        
        # Detect anomalies using gradient
        gradient = np.gradient(depth_gray.astype(float))
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Threshold for anomaly detection
        anomaly_threshold = 20
        anomalies = gradient_magnitude > anomaly_threshold
        
        # Save
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f"carla_simulation_output/depth_{timestamp}.png", depth_gray)
        cv2.imwrite(f"carla_simulation_output/depth_anomaly_{timestamp}.png", 
                   (anomalies * 255).astype(np.uint8))
    
    def spawn_simulated_defects(self):
        """
        Spawn props to simulate road defects
        Uses small static objects to represent potholes and debris
        """
        vehicle_location = self.vehicle.get_location()
        forward = self.vehicle.get_transform().get_forward_vector()
        
        defect_types = [
            'static.prop.dirtdebris01',
            'static.prop.dirtdebris02',
            'static.prop.dirtdebris03',
            'static.prop.barrel',
            'static.prop.box03'
        ]
        
        for i in range(10):
            try:
                # Random position ahead
                offset = random.uniform(20, 100)
                lateral = random.uniform(-3, 3)
                
                defect_bp = self.blueprint_library.find(random.choice(defect_types))
                
                transform = carla.Transform(
                    carla.Location(
                        x=vehicle_location.x + forward.x * offset,
                        y=vehicle_location.y + forward.y * offset + lateral,
                        z=vehicle_location.z - 0.3  # Slightly below road surface
                    ),
                    carla.Rotation(yaw=random.uniform(0, 360))
                )
                
                defect = self.world.spawn_actor(defect_bp, transform)
                self.defects.append({
                    'actor': defect,
                    'type': 'simulated_pothole',
                    'location': transform.location
                })
                
            except:
                continue
        
        print(f"✓ Spawned {len(self.defects)} simulated road defects")
        return self.defects
    
    def analyze_imu_for_bumps(self, imu_data_list):
        """
        Analyze IMU data to detect speed bumps and potholes
        Sudden vertical acceleration indicates bump
        """
        bumps_detected = []
        
        for i in range(1, len(imu_data_list)):
            current = imu_data_list[i]
            previous = imu_data_list[i-1]
            
            # Calculate z-axis acceleration change
            z_change = abs(current['acceleration']['z'] - 
                          previous['acceleration']['z'])
            
            # Threshold for bump detection
            if z_change > 5.0:  # Significant vertical acceleration
                bumps_detected.append({
                    'timestamp': current['timestamp'],
                    'z_acceleration_change': z_change,
                    'type': 'speed_bump' if z_change > 8.0 else 'pothole'
                })
        
        return bumps_detected


def run_road_defect_scenario():
    """Main function to run road defect detection scenario"""
    print("="*60)
    print("ROAD DEFECT DETECTION SCENARIO")
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
        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        vehicle.set_autopilot(True)
        
        print("✓ Vehicle spawned")
        
        # Create scenario
        scenario = RoadDefectScenario(world, vehicle)
        
        # Attach specialized cameras
        semantic_camera = scenario.create_semantic_camera()
        depth_camera = scenario.create_depth_camera()
        
        print("✓ Cameras attached")
        
        # Spawn defects
        scenario.spawn_simulated_defects()
        
        # Run simulation
        print("\nRunning simulation for 60 seconds...")
        for i in range(1200):  # 60 seconds at 20 FPS
            world.tick()
            
            if i % 100 == 0:
                print(f"  Progress: {i/20:.1f}s / 60s")
        
        print("\n✓ Scenario completed!")
        
        # Cleanup
        semantic_camera.destroy()
        depth_camera.destroy()
        vehicle.destroy()
        
        for defect in scenario.defects:
            if defect['actor'].is_alive:
                defect['actor'].destroy()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    run_road_defect_scenario()