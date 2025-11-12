# """GPS location utilities using Android Location API directly"""
# from kivy.utils import platform
# from kivy.clock import Clock

# if platform == 'android':
#     from jnius import autoclass, PythonJavaClass, java_method, cast
#     from android.permissions import request_permissions, Permission, check_permission
    
#     # Android classes - renamed to avoid conflict
#     PythonActivity = autoclass('org.kivy.android.PythonActivity')
#     AndroidLocationManager = autoclass('android.location.LocationManager')
#     Context = autoclass('android.content.Context')
    
#     def request_location_permissions():
#         """Request Android location permissions"""
#         request_permissions([
#             Permission.ACCESS_FINE_LOCATION,
#             Permission.ACCESS_COARSE_LOCATION
#         ])
#         return check_permission(Permission.ACCESS_FINE_LOCATION)
# else:
#     def request_location_permissions():
#         return True

# class GPSLocationManager:
#     """Manages GPS location tracking"""
    
#     def __init__(self):
#         self.current_lat = None
#         self.current_lon = None
#         self.callbacks = []
#         self.android_location_manager = None
#         self.location_listener = None
        
#     def start(self, on_location=None):
#         """Start GPS tracking"""
#         if on_location:
#             self.callbacks.append(on_location)
        
#         if platform != 'android':
#             print("Not on Android - GPS disabled")
#             return
        
#         # Request permissions
#         request_location_permissions()
        
#         # Delay GPS start to allow permission grant
#         Clock.schedule_once(lambda dt: self._start_android_gps(), 1)
    
#     def _start_android_gps(self):
#         """Start Android GPS using LocationManager"""
#         try:
#             activity = PythonActivity.mActivity
#             self.android_location_manager = cast(
#                 AndroidLocationManager,
#                 activity.getSystemService(Context.LOCATION_SERVICE)
#             )
            
#             # Create location listener
#             class LocationListener(PythonJavaClass):
#                 __javainterfaces__ = ['android/location/LocationListener']
                
#                 @java_method('(Landroid/location/Location;)V')
#                 def onLocationChanged(self, location):
#                     lat = location.getLatitude()
#                     lon = location.getLongitude()
#                     accuracy = location.getAccuracy()
                    
#                     print(f"✓ GPS Location: {lat:.6f}, {lon:.6f} (±{accuracy:.1f}m)")
                    
#                     self.parent.current_lat = lat
#                     self.parent.current_lon = lon
                    
#                     # Notify callbacks
#                     for callback in self.parent.callbacks:
#                         try:
#                             callback(lat, lon)
#                         except Exception as e:
#                             print(f"Callback error: {e}")
                
#                 @java_method('(Ljava/lang/String;)V')
#                 def onProviderEnabled(self, provider):
#                     print(f"✓ GPS Provider enabled: {provider}")
                
#                 @java_method('(Ljava/lang/String;)V')
#                 def onProviderDisabled(self, provider):
#                     print(f"✗ GPS Provider disabled: {provider}")
                
#                 @java_method('(Ljava/lang/String;ILandroid/os/Bundle;)V')
#                 def onStatusChanged(self, provider, status, extras):
#                     status_map = {0: 'OUT_OF_SERVICE', 1: 'TEMPORARILY_UNAVAILABLE', 2: 'AVAILABLE'}
#                     print(f"GPS Status: {provider} - {status_map.get(status, 'UNKNOWN')}")
            
#             self.location_listener = LocationListener()
#             self.location_listener.parent = self
            
#             # Request location updates from GPS and Network providers
#             providers = ['gps', 'network']
            
#             for provider in providers:
#                 if self.android_location_manager.isProviderEnabled(provider):
#                     print(f"✓ Requesting location updates from: {provider}")
#                     self.android_location_manager.requestLocationUpdates(
#                         provider,
#                         1000,  # minTime: 1 second
#                         0,     # minDistance: 0 meters
#                         self.location_listener
#                     )
                    
#                     # Get last known location immediately
#                     last_location = self.android_location_manager.getLastKnownLocation(provider)
#                     if last_location:
#                         lat = last_location.getLatitude()
#                         lon = last_location.getLongitude()
#                         self.current_lat = lat
#                         self.current_lon = lon
#                         print(f"✓ Got last known location from {provider}: {lat:.6f}, {lon:.6f}")
                        
#                         # Notify callbacks
#                         for callback in self.callbacks:
#                             try:
#                                 callback(lat, lon)
#                             except Exception as e:
#                                 print(f"Callback error: {e}")
#                 else:
#                     print(f"✗ Provider not enabled: {provider}")
            
#             print("✓ GPS started successfully")
            
#         except Exception as e:
#             print(f"✗ GPS start error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def stop(self):
#         """Stop GPS tracking"""
#         if platform != 'android':
#             return
            
#         try:
#             if self.android_location_manager and self.location_listener:
#                 self.android_location_manager.removeUpdates(self.location_listener)
#                 print("GPS stopped")
#         except Exception as e:
#             print(f"GPS stop error: {e}")
    
#     def get_location(self):
#         """Get current location"""
#         return self.current_lat, self.current_lon


# # For backward compatibility
# LocationManager = GPSLocationManager


"""GPS location using Plyer (most reliable for Kivy)"""
from kivy.utils import platform
from kivy.clock import Clock

if platform == 'android':
    from plyer import gps
    from android.permissions import request_permissions, Permission, check_permission
    
    def request_location_permissions():
        """Request location permissions"""
        try:
            if not check_permission(Permission.ACCESS_FINE_LOCATION):
                request_permissions([
                    Permission.ACCESS_FINE_LOCATION,
                    Permission.ACCESS_COARSE_LOCATION
                ])
                return False
            return True
        except:
            return False
else:
    class MockGPS:
        def configure(self, *args, **kwargs): pass
        def start(self, *args, **kwargs): pass
        def stop(self): pass
    gps = MockGPS()
    def request_location_permissions(): return True

class LocationManager:
    """GPS Location Manager using Plyer"""
    
    def __init__(self):
        self.current_lat = None
        self.current_lon = None
        self.callbacks = []
        self.gps_started = False
        
    def start(self, on_location=None):
        """Start GPS"""
        if on_location:
            self.callbacks.append(on_location)
        
        if platform != 'android':
            print("Not on Android - GPS disabled")
            return
        
        # Request permissions
        request_location_permissions()
        
        # Delay start
        Clock.schedule_once(lambda dt: self._start_gps(), 2)
    
    def _start_gps(self):
        """Internal GPS start"""
        if self.gps_started:
            return
        
        try:
            print("Starting Plyer GPS...")
            gps.configure(
                on_location=self._on_location,
                on_status=self._on_status
            )
            gps.start(minTime=1000, minDistance=0)
            self.gps_started = True
            print("✓ GPS started")
        except Exception as e:
            print(f"✗ GPS error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_location(self, **kwargs):
        """Location callback"""
        lat = kwargs.get('lat')
        lon = kwargs.get('lon')
        
        if lat and lon:
            self.current_lat = lat
            self.current_lon = lon
            print(f"✓ GPS: {lat:.6f}, {lon:.6f}")
            
            for cb in self.callbacks:
                try:
                    cb(lat, lon)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def _on_status(self, stype, status):
        """Status callback"""
        print(f"GPS {stype}: {status}")
        
        if stype == 'provider-disabled':
            print("✗ GPS disabled in settings!")
    
    def stop(self):
        """Stop GPS"""
        if platform == 'android' and self.gps_started:
            try:
                gps.stop()
                self.gps_started = False
            except:
                pass
    
    def get_location(self):
        """Get current location"""
        return self.current_lat, self.current_lon
