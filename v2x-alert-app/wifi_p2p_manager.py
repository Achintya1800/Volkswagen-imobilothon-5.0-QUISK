


# # """WiFi Direct P2P Manager for hazard alert transmission"""
# # import json
# # import socket
# # import threading
# # from kivy.utils import platform

# # # Import Android modules (only imported when running on Android device)
# # if platform == 'android':
# #     from jnius import autoclass, cast, PythonJavaClass, java_method
# #     from android.permissions import request_permissions, Permission
    
# #     # Java classes
# #     WifiP2pManager = autoclass('android.net.wifi.p2p.WifiP2pManager')
# #     WifiP2pConfig = autoclass('android.net.wifi.p2p.WifiP2pConfig')
# #     WifiP2pDnsSdServiceInfo = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceInfo')
# #     WifiP2pDnsSdServiceRequest = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceRequest')
# #     Context = autoclass('android.content.Context')
# #     PythonActivity = autoclass('org.kivy.android.PythonActivity')

# # def request_wifi_permissions():
# #     """Request WiFi P2P permissions - REAL Android implementation"""
# #     if platform != 'android':
# #         print("Desktop mode - WiFi P2P not available")
# #         return
        
# #     request_permissions([
# #         Permission.ACCESS_WIFI_STATE,
# #         Permission.CHANGE_WIFI_STATE,
# #         Permission.ACCESS_FINE_LOCATION,
# #         Permission.NEARBY_WIFI_DEVICES
# #     ])

# # class WiFiP2PManager:
# #     """Manages WiFi Direct P2P connections for hazard alerts"""
    
# #     def __init__(self):
# #         self.is_android = platform == 'android'
# #         self.manager = None
# #         self.channel = None
# #         self.receiver = None
# #         self.service_info = None
# #         self.is_group_owner = False
# #         self.on_hazard_received = None
        
# #         if self.is_android:
# #             self._init_android()
# #         else:
# #             print("WARNING: Not on Android - WiFi P2P disabled")
    
# #     def _init_android(self):
# #         """Initialize Android WiFi P2P manager - REAL Android code"""
# #         request_wifi_permissions()
        
# #         try:
# #             activity = PythonActivity.mActivity
# #             self.manager = cast(
# #                 WifiP2pManager,
# #                 activity.getSystemService(Context.WIFI_P2P_SERVICE)
# #             )
# #             self.channel = self.manager.initialize(
# #                 activity,
# #                 activity.getMainLooper(),
# #                 None
# #             )
# #             print("✓ WiFi P2P Manager initialized successfully")
# #         except Exception as e:
# #             print(f"✗ WiFi P2P init error: {e}")
# #             import traceback
# #             traceback.print_exc()
    
# #     def start_service_as_sender(self, hazard_data):
# #         """
# #         Start WiFi Direct service as hazard alert sender (Group Owner)
# #         REAL Android WiFi Direct implementation - NO MOCK
        
# #         Args:
# #             hazard_data: dict containing hazard information
# #         """
# #         if not self.is_android:
# #             print("Desktop mode - Cannot broadcast WiFi P2P")
# #             return
        
# #         if not self.manager:
# #             print("✗ WiFi P2P Manager not initialized")
# #             return
        
# #         try:
# #             # Create service info with hazard metadata
# #             service_map = {
# #                 "hazard_type": str(hazard_data.get("type", "unknown")),
# #                 "latitude": str(hazard_data.get("latitude", 0.0)),
# #                 "longitude": str(hazard_data.get("longitude", 0.0)),
# #                 "severity": str(hazard_data.get("severity", "medium")),
# #                 "timestamp": str(hazard_data.get("timestamp", 0))
# #             }
            
# #             print(f"Creating WiFi Direct service with data: {service_map}")
            
# #             # Create DNS-SD service
# #             self.service_info = WifiP2pDnsSdServiceInfo.newInstance(
# #                 "_hazardalert",      # Instance name
# #                 "_presence._tcp",     # Service type
# #                 service_map          # TXT record map
# #             )
            
# #             # Add local service
# #             class AddServiceListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
# #                 @java_method('()V')
# #                 def onSuccess(self):
# #                     print("✓ WiFi Direct service registered successfully")
# #                     self.parent.is_group_owner = True
# #                     # Start socket server
# #                     self.parent._start_socket_server(hazard_data)
                
# #                 @java_method('(I)V')
# #                 def onFailure(self, reason):
# #                     print(f"✗ Service registration failed - Error code: {reason}")
# #                     error_msgs = {
# #                         0: "ERROR",
# #                         1: "P2P_UNSUPPORTED",
# #                         2: "BUSY"
# #                     }
# #                     print(f"  Reason: {error_msgs.get(reason, 'UNKNOWN')}")
            
# #             listener = AddServiceListener()
# #             listener.parent = self
# #             listener.hazard_data = hazard_data
            
# #             self.manager.addLocalService(
# #                 self.channel,
# #                 self.service_info,
# #                 listener
# #             )
            
# #             print("📡 Broadcasting hazard alert via WiFi Direct...")
            
# #         except Exception as e:
# #             print(f"✗ Error starting sender service: {e}")
# #             import traceback
# #             traceback.print_exc()
    
# #     def _start_socket_server(self, hazard_data):
# #         """Start TCP socket server to send full hazard data - REAL implementation"""
# #         def server_thread():
# #             try:
# #                 server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #                 server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
# #                 # Group Owner uses 192.168.49.1
# #                 server.bind(('0.0.0.0', 8888))
# #                 server.listen(5)
                
# #                 print("✓ Socket server listening on port 8888 (Group Owner)")
                
# #                 while True:
# #                     client, addr = server.accept()
# #                     print(f"✓ Client connected from {addr}")
                    
# #                     # Send hazard JSON
# #                     message = json.dumps(hazard_data).encode('utf-8')
# #                     client.send(message)
# #                     print(f"✓ Sent hazard data to {addr}")
                    
# #                     client.close()
                    
# #             except Exception as e:
# #                 print(f"✗ Socket server error: {e}")
# #                 import traceback
# #                 traceback.print_exc()
        
# #         threading.Thread(target=server_thread, daemon=True).start()
    
# #     def start_discovery_as_receiver(self, on_hazard_callback):
# #         """
# #         Start WiFi Direct service discovery as receiver
# #         REAL Android WiFi Direct implementation - NO MOCK
        
# #         Args:
# #             on_hazard_callback: Function called when hazard is discovered
# #         """
# #         if not self.is_android:
# #             print("Desktop mode - Cannot discover WiFi P2P services")
# #             return
        
# #         if not self.manager:
# #             print("✗ WiFi P2P Manager not initialized")
# #             return
        
# #         self.on_hazard_received = on_hazard_callback
        
# #         try:
# #             # DNS-SD TXT record listener
# #             class TxtRecordListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdTxtRecordListener']
                
# #                 @java_method('(Ljava/lang/String;Ljava/util/Map;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
# #                 def onDnsSdTxtRecordAvailable(self, fullDomain, txtRecord, device):
# #                     print(f"✓ WiFi Direct service discovered: {fullDomain}")
# #                     print(f"  Device: {device.deviceName}")
                    
# #                     try:
# #                         # Extract hazard metadata
# #                         hazard_metadata = {
# #                             "type": txtRecord.get("hazard_type"),
# #                             "latitude": float(txtRecord.get("latitude")),
# #                             "longitude": float(txtRecord.get("longitude")),
# #                             "severity": txtRecord.get("severity"),
# #                             "device": device
# #                         }
                        
# #                         print(f"  Hazard: {hazard_metadata['type']} at {hazard_metadata['latitude']}, {hazard_metadata['longitude']}")
                        
# #                         # Check geofence and connect
# #                         self.parent._handle_discovered_hazard(hazard_metadata)
# #                     except Exception as e:
# #                         print(f"✗ Error processing discovered service: {e}")
            
# #             # DNS-SD service listener
# #             class ServiceListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdServiceResponseListener']
                
# #                 @java_method('(Ljava/lang/String;Ljava/lang/String;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
# #                 def onDnsSdServiceAvailable(self, instanceName, registrationType, device):
# #                     print(f"✓ Service available: {instanceName} ({registrationType})")
# #                     print(f"  Device: {device.deviceName} [{device.deviceAddress}]")
            
# #             txt_listener = TxtRecordListener()
# #             txt_listener.parent = self
            
# #             service_listener = ServiceListener()
            
# #             # Set response listeners
# #             self.manager.setDnsSdResponseListeners(
# #                 self.channel,
# #                 service_listener,
# #                 txt_listener
# #             )
            
# #             # Create service request
# #             service_request = WifiP2pDnsSdServiceRequest.newInstance()
            
# #             # Add service request
# #             class RequestListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
# #                 @java_method('()V')
# #                 def onSuccess(self):
# #                     print("✓ Service request added successfully")
                
# #                 @java_method('(I)V')
# #                 def onFailure(self, reason):
# #                     print(f"✗ Service request failed - Error code: {reason}")
            
# #             self.manager.addServiceRequest(
# #                 self.channel,
# #                 service_request,
# #                 RequestListener()
# #             )
            
# #             # Start discovery
# #             class DiscoveryListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
# #                 @java_method('()V')
# #                 def onSuccess(self):
# #                     print("✓ WiFi Direct service discovery started successfully")
                
# #                 @java_method('(I)V')
# #                 def onFailure(self, reason):
# #                     print(f"✗ Discovery start failed - Error code: {reason}")
            
# #             self.manager.discoverServices(
# #                 self.channel,
# #                 DiscoveryListener()
# #             )
            
# #             print("📡 Scanning for WiFi Direct hazard alerts...")
            
# #         except Exception as e:
# #             print(f"✗ Error starting discovery: {e}")
# #             import traceback
# #             traceback.print_exc()
    
# #     def _handle_discovered_hazard(self, hazard_metadata):
# #         """Handle discovered hazard - check geofence and connect"""
# #         from geofence import is_within_geofence
# #         from utils.location import LocationManager
        
# #         # Get current location
# #         loc_mgr = LocationManager()
# #         current_lat, current_lon = loc_mgr.get_location()
        
# #         if current_lat is None or current_lon is None:
# #             print("⚠ Warning: GPS not ready - cannot check geofence")
# #             print("  Connecting anyway for demo purposes")
# #             # For demo, connect even without GPS
# #             self._connect_to_sender(hazard_metadata["device"])
# #             return
        
# #         # Check geofence
# #         within_fence, distance = is_within_geofence(
# #             current_lat, current_lon,
# #             hazard_metadata["latitude"],
# #             hazard_metadata["longitude"],
# #             radius_m=500  # 500 meter geofence
# #         )
        
# #         print(f"Distance to hazard: {distance:.1f}m")
        
# #         if within_fence:
# #             print(f"⚠ ENTERING GEOFENCE - Connecting to receive alert")
# #             self._connect_to_sender(hazard_metadata["device"])
# #         else:
# #             print(f"Outside geofence ({distance:.1f}m > 500m) - NOT connecting")
    
# #     def _connect_to_sender(self, device):
# #         """Connect to sender device to receive full hazard data"""
# #         try:
# #             config = WifiP2pConfig()
# #             config.deviceAddress = device.deviceAddress
            
# #             class ConnectListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
# #                 @java_method('()V')
# #                 def onSuccess(self):
# #                     print("✓ Connected to sender device")
# #                     # Get group info and connect via socket
# #                     self.parent._get_connection_info()
                
# #                 @java_method('(I)V')
# #                 def onFailure(self, reason):
# #                     print(f"✗ Connection failed - Error code: {reason}")
            
# #             listener = ConnectListener()
# #             listener.parent = self
            
# #             self.manager.connect(self.channel, config, listener)
# #             print(f"Connecting to device: {device.deviceName}...")
            
# #         except Exception as e:
# #             print(f"✗ Connection error: {e}")
# #             import traceback
# #             traceback.print_exc()
    
# #     def _get_connection_info(self):
# #         """Get connection info and receive hazard data"""
# #         try:
# #             class ConnectionInfoListener(PythonJavaClass):
# #                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ConnectionInfoListener']
                
# #                 @java_method('(Landroid/net/wifi/p2p/WifiP2pInfo;)V')
# #                 def onConnectionInfoAvailable(self, info):
# #                     if info.groupFormed:
# #                         go_address = info.groupOwnerAddress.getHostAddress()
# #                         print(f"✓ Group formed - Group Owner IP: {go_address}")
# #                         # Receive hazard data
# #                         self.parent._receive_hazard_data(go_address)
# #                     else:
# #                         print("⚠ Group not formed yet")
            
# #             listener = ConnectionInfoListener()
# #             listener.parent = self
            
# #             self.manager.requestConnectionInfo(self.channel, listener)
            
# #         except Exception as e:
# #             print(f"✗ Connection info error: {e}")
# #             import traceback
# #             traceback.print_exc()
    
# #     def _receive_hazard_data(self, go_address):
# #         """Receive full hazard data via TCP socket"""
# #         def receive_thread():
# #             try:
# #                 print(f"Connecting to socket server at {go_address}:8888...")
# #                 client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #                 client.settimeout(10)  # 10 second timeout
# #                 client.connect((go_address, 8888))
                
# #                 print("✓ Connected to socket server")
# #                 data = client.recv(4096).decode('utf-8')
# #                 hazard = json.loads(data)
                
# #                 print(f"✓ Received full hazard data: {hazard}")
                
# #                 # Trigger callback
# #                 if self.on_hazard_received:
# #                     self.on_hazard_received(hazard)
                
# #                 client.close()
                
# #             except socket.timeout:
# #                 print("✗ Socket connection timeout")
# #             except Exception as e:
# #                 print(f"✗ Receive error: {e}")
# #                 import traceback
# #                 traceback.print_exc()
        
# #         threading.Thread(target=receive_thread, daemon=True).start()
    
# #     def stop(self):
# #         """Stop WiFi P2P services"""
# #         if not self.is_android:
# #             return
        
# #         try:
# #             if self.service_info and self.manager:
# #                 self.manager.removeLocalService(self.channel, self.service_info, None)
# #             if self.manager:
# #                 self.manager.stopPeerDiscovery(self.channel, None)
# #             print("✓ WiFi P2P stopped")
# #         except Exception as e:
# #             print(f"WiFi P2P stop error: {e}")




# """WiFi Direct P2P Manager for hazard alert transmission with error handling"""
# import json
# import socket
# import threading
# from kivy.utils import platform
# from kivy.clock import Clock

# # Import Android modules (only imported when running on Android device)
# if platform == 'android':
#     try:
#         from jnius import autoclass, cast, PythonJavaClass, java_method
#         from android.permissions import request_permissions, Permission
        
#         # Java classes
#         WifiP2pManager = autoclass('android.net.wifi.p2p.WifiP2pManager')
#         WifiP2pConfig = autoclass('android.net.wifi.p2p.WifiP2pConfig')
#         WifiP2pDnsSdServiceInfo = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceInfo')
#         WifiP2pDnsSdServiceRequest = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceRequest')
#         Context = autoclass('android.content.Context')
#         PythonActivity = autoclass('org.kivy.android.PythonActivity')
        
#         JNIUS_AVAILABLE = True
#     except Exception as e:
#         print(f"⚠ Warning: Could not import Android modules: {e}")
#         JNIUS_AVAILABLE = False
# else:
#     JNIUS_AVAILABLE = False

# def request_wifi_permissions():
#     """Request WiFi P2P permissions - REAL Android implementation"""
#     if platform != 'android' or not JNIUS_AVAILABLE:
#         print("Desktop mode or jnius unavailable - WiFi P2P not available")
#         return
    
#     try:
#         request_permissions([
#             Permission.ACCESS_WIFI_STATE,
#             Permission.CHANGE_WIFI_STATE,
#             Permission.ACCESS_FINE_LOCATION,
#             Permission.ACCESS_COARSE_LOCATION,
#             Permission.NEARBY_WIFI_DEVICES
#         ])
#     except Exception as e:
#         print(f"Permission request error: {e}")

# class WiFiP2PManager:
#     """Manages WiFi Direct P2P connections for hazard alerts"""
    
#     def __init__(self):
#         self.is_android = platform == 'android' and JNIUS_AVAILABLE
#         self.manager = None
#         self.channel = None
#         self.receiver = None
#         self.service_info = None
#         self.is_group_owner = False
#         self.on_hazard_received = None
        
#         if self.is_android:
#             self._init_android()
#         else:
#             print("WARNING: Not on Android or jnius unavailable - WiFi P2P disabled")
    
#     def _init_android(self):
#         """Initialize Android WiFi P2P manager - REAL Android code"""
#         try:
#             request_wifi_permissions()
            
#             activity = PythonActivity.mActivity
#             self.manager = cast(
#                 WifiP2pManager,
#                 activity.getSystemService(Context.WIFI_P2P_SERVICE)
#             )
            
#             if self.manager:
#                 self.channel = self.manager.initialize(
#                     activity,
#                     activity.getMainLooper(),
#                     None
#                 )
#                 print("✓ WiFi P2P Manager initialized successfully")
#             else:
#                 print("✗ WiFi P2P Manager is null - service not available")
                
#         except Exception as e:
#             print(f"✗ WiFi P2P init error: {e}")
#             import traceback
#             traceback.print_exc()
#             self.manager = None
    
#     def start_service_as_sender(self, hazard_data):
#         """
#         Start WiFi Direct service as hazard alert sender (Group Owner)
#         """
#         if not self.is_android:
#             print("📡 DEMO: Broadcasting hazard (Desktop mode)")
#             return
        
#         if not self.manager:
#             print("✗ WiFi P2P Manager not initialized - Cannot broadcast")
#             return
        
#         try:
#             # Create service info with hazard metadata
#             service_map = {
#                 "hazard_type": str(hazard_data.get("type", "unknown")),
#                 "latitude": str(hazard_data.get("latitude", 0.0)),
#                 "longitude": str(hazard_data.get("longitude", 0.0)),
#                 "severity": str(hazard_data.get("severity", "medium")),
#                 "timestamp": str(hazard_data.get("timestamp", 0))
#             }
            
#             print(f"Creating WiFi Direct service with data: {service_map}")
            
#             # Create DNS-SD service
#             self.service_info = WifiP2pDnsSdServiceInfo.newInstance(
#                 "_hazardalert",
#                 "_presence._tcp",
#                 service_map
#             )
            
#             # Add local service
#             class AddServiceListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
#                 @java_method('()V')
#                 def onSuccess(self):
#                     print("✓ WiFi Direct service registered successfully")
#                     self.parent.is_group_owner = True
#                     self.parent._start_socket_server(self.hazard_data)
                
#                 @java_method('(I)V')
#                 def onFailure(self, reason):
#                     print(f"✗ Service registration failed - Error code: {reason}")
#                     error_msgs = {0: "ERROR", 1: "P2P_UNSUPPORTED", 2: "BUSY"}
#                     print(f"  Reason: {error_msgs.get(reason, 'UNKNOWN')}")
            
#             listener = AddServiceListener()
#             listener.parent = self
#             listener.hazard_data = hazard_data
            
#             self.manager.addLocalService(
#                 self.channel,
#                 self.service_info,
#                 listener
#             )
            
#             print("📡 Broadcasting hazard alert via WiFi Direct...")
            
#         except Exception as e:
#             print(f"✗ Error starting sender service: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _start_socket_server(self, hazard_data):
#         """Start TCP socket server to send full hazard data"""
#         def server_thread():
#             try:
#                 server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#                 server.bind(('0.0.0.0', 8888))
#                 server.listen(5)
                
#                 print("✓ Socket server listening on port 8888")
                
#                 while True:
#                     client, addr = server.accept()
#                     print(f"✓ Client connected from {addr}")
                    
#                     message = json.dumps(hazard_data).encode('utf-8')
#                     client.send(message)
#                     print(f"✓ Sent hazard data to {addr}")
#                     client.close()
                    
#             except Exception as e:
#                 print(f"✗ Socket server error: {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         threading.Thread(target=server_thread, daemon=True).start()
    
#     def start_discovery_as_receiver(self, on_hazard_callback):
#         """
#         Start WiFi Direct service discovery as receiver
#         """
#         if not self.is_android:
#             print("📡 DEMO: Scanning for hazards (Desktop mode)")
#             # Simulate receiving alert after 3 seconds for demo
#             Clock.schedule_once(lambda dt: self._simulate_demo_alert(on_hazard_callback), 3)
#             return
        
#         if not self.manager:
#             print("✗ WiFi P2P Manager not initialized - Cannot scan")
#             return
        
#         self.on_hazard_received = on_hazard_callback
        
#         try:
#             # Request WiFi permissions again before starting
#             request_wifi_permissions()
            
#             # Add 1 second delay before starting discovery
#             Clock.schedule_once(lambda dt: self._start_discovery(), 1)
            
#         except Exception as e:
#             print(f"✗ Error starting discovery: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _start_discovery(self):
#         """Internal method to start discovery after delay"""
#         try:
#             # DNS-SD TXT record listener
#             class TxtRecordListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdTxtRecordListener']
                
#                 @java_method('(Ljava/lang/String;Ljava/util/Map;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
#                 def onDnsSdTxtRecordAvailable(self, fullDomain, txtRecord, device):
#                     print(f"✓ WiFi Direct service discovered: {fullDomain}")
#                     print(f"  Device: {device.deviceName}")
                    
#                     try:
#                         hazard_metadata = {
#                             "type": txtRecord.get("hazard_type"),
#                             "latitude": float(txtRecord.get("latitude")),
#                             "longitude": float(txtRecord.get("longitude")),
#                             "severity": txtRecord.get("severity"),
#                             "device": device
#                         }
                        
#                         print(f"  Hazard: {hazard_metadata['type']} at {hazard_metadata['latitude']}, {hazard_metadata['longitude']}")
#                         self.parent._handle_discovered_hazard(hazard_metadata)
#                     except Exception as e:
#                         print(f"✗ Error processing discovered service: {e}")
            
#             # DNS-SD service listener
#             class ServiceListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdServiceResponseListener']
                
#                 @java_method('(Ljava/lang/String;Ljava/lang/String;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
#                 def onDnsSdServiceAvailable(self, instanceName, registrationType, device):
#                     print(f"✓ Service available: {instanceName} ({registrationType})")
#                     print(f"  Device: {device.deviceName} [{device.deviceAddress}]")
            
#             txt_listener = TxtRecordListener()
#             txt_listener.parent = self
            
#             service_listener = ServiceListener()
            
#             # Set response listeners
#             self.manager.setDnsSdResponseListeners(
#                 self.channel,
#                 service_listener,
#                 txt_listener
#             )
            
#             # Create service request
#             service_request = WifiP2pDnsSdServiceRequest.newInstance()
            
#             # Add service request
#             class RequestListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
#                 @java_method('()V')
#                 def onSuccess(self):
#                     print("✓ Service request added successfully")
                
#                 @java_method('(I)V')
#                 def onFailure(self, reason):
#                     print(f"✗ Service request failed - Error code: {reason}")
            
#             self.manager.addServiceRequest(
#                 self.channel,
#                 service_request,
#                 RequestListener()
#             )
            
#             # Start discovery
#             class DiscoveryListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
#                 @java_method('()V')
#                 def onSuccess(self):
#                     print("✓ WiFi Direct service discovery started successfully")
                
#                 @java_method('(I)V')
#                 def onFailure(self, reason):
#                     print(f"✗ Discovery start failed - Error code: {reason}")
            
#             self.manager.discoverServices(
#                 self.channel,
#                 DiscoveryListener()
#             )
            
#             print("📡 Scanning for WiFi Direct hazard alerts...")
            
#         except Exception as e:
#             print(f"✗ Discovery error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _simulate_demo_alert(self, callback):
#         """Simulate receiving an alert for demo purposes"""
#         print("✓ DEMO: Simulated hazard alert received!")
        
#         demo_hazard = {
#             "type": "pothole",
#             "latitude": 19.1246278,
#             "longitude": 72.8373842,
#             "severity": "high",
#             "confidence": 0.92,
#             "timestamp": 1699824000
#         }
        
#         if callback:
#             callback(demo_hazard)
    
#     def _handle_discovered_hazard(self, hazard_metadata):
#         """Handle discovered hazard - check geofence and connect"""
#         try:
#             from geofence import is_within_geofence
#             from utils.location import LocationManager
            
#             loc_mgr = LocationManager()
#             current_lat, current_lon = loc_mgr.get_location()
            
#             if current_lat is None or current_lon is None:
#                 print("⚠ Warning: GPS not ready - connecting anyway for demo")
#                 self._connect_to_sender(hazard_metadata["device"])
#                 return
            
#             within_fence, distance = is_within_geofence(
#                 current_lat, current_lon,
#                 hazard_metadata["latitude"],
#                 hazard_metadata["longitude"],
#                 radius_m=500
#             )
            
#             print(f"Distance to hazard: {distance:.1f}m")
            
#             if within_fence:
#                 print(f"⚠ ENTERING GEOFENCE - Connecting to receive alert")
#                 self._connect_to_sender(hazard_metadata["device"])
#             else:
#                 print(f"Outside geofence ({distance:.1f}m > 500m) - NOT connecting")
                
#         except Exception as e:
#             print(f"Error handling discovered hazard: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _connect_to_sender(self, device):
#         """Connect to sender device to receive full hazard data"""
#         try:
#             config = WifiP2pConfig()
#             config.deviceAddress = device.deviceAddress
            
#             class ConnectListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
#                 @java_method('()V')
#                 def onSuccess(self):
#                     print("✓ Connected to sender device")
#                     self.parent._get_connection_info()
                
#                 @java_method('(I)V')
#                 def onFailure(self, reason):
#                     print(f"✗ Connection failed - Error code: {reason}")
            
#             listener = ConnectListener()
#             listener.parent = self
            
#             self.manager.connect(self.channel, config, listener)
#             print(f"Connecting to device: {device.deviceName}...")
            
#         except Exception as e:
#             print(f"✗ Connection error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _get_connection_info(self):
#         """Get connection info and receive hazard data"""
#         try:
#             class ConnectionInfoListener(PythonJavaClass):
#                 __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ConnectionInfoListener']
                
#                 @java_method('(Landroid/net/wifi/p2p/WifiP2pInfo;)V')
#                 def onConnectionInfoAvailable(self, info):
#                     if info.groupFormed:
#                         go_address = info.groupOwnerAddress.getHostAddress()
#                         print(f"✓ Group formed - Group Owner IP: {go_address}")
#                         self.parent._receive_hazard_data(go_address)
#                     else:
#                         print("⚠ Group not formed yet")
            
#             listener = ConnectionInfoListener()
#             listener.parent = self
            
#             self.manager.requestConnectionInfo(self.channel, listener)
            
#         except Exception as e:
#             print(f"✗ Connection info error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     def _receive_hazard_data(self, go_address):
#         """Receive full hazard data via TCP socket"""
#         def receive_thread():
#             try:
#                 print(f"Connecting to socket server at {go_address}:8888...")
#                 client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 client.settimeout(10)
#                 client.connect((go_address, 8888))
                
#                 print("✓ Connected to socket server")
#                 data = client.recv(4096).decode('utf-8')
#                 hazard = json.loads(data)
                
#                 print(f"✓ Received full hazard data: {hazard}")
                
#                 if self.on_hazard_received:
#                     self.on_hazard_received(hazard)
                
#                 client.close()
                
#             except socket.timeout:
#                 print("✗ Socket connection timeout")
#             except Exception as e:
#                 print(f"✗ Receive error: {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         threading.Thread(target=receive_thread, daemon=True).start()
    
#     def stop(self):
#         """Stop WiFi P2P services"""
#         if not self.is_android or not self.manager:
#             return
        
#         try:
#             if self.service_info:
#                 self.manager.removeLocalService(self.channel, self.service_info, None)
#             self.manager.stopPeerDiscovery(self.channel, None)
#             print("✓ WiFi P2P stopped")
#         except Exception as e:
#             print(f"WiFi P2P stop error: {e}")


"""WiFi Direct P2P Manager with maximum crash protection"""
import json
import socket
import threading
from kivy.utils import platform
from kivy.clock import Clock

# Safely import Android modules
ANDROID_AVAILABLE = False
if platform == 'android':
    try:
        from jnius import autoclass, cast, PythonJavaClass, java_method
        from android.permissions import request_permissions, Permission
        
        WifiP2pManager = autoclass('android.net.wifi.p2p.WifiP2pManager')
        WifiP2pConfig = autoclass('android.net.wifi.p2p.WifiP2pConfig')
        WifiP2pDnsSdServiceInfo = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceInfo')
        WifiP2pDnsSdServiceRequest = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceRequest')
        Context = autoclass('android.content.Context')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        
        ANDROID_AVAILABLE = True
        print("✓ Android WiFi P2P modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Android modules: {e}")
        ANDROID_AVAILABLE = False

def request_wifi_permissions():
    """Request WiFi P2P permissions"""
    if not ANDROID_AVAILABLE:
        return False
    
    try:
        request_permissions([
            Permission.ACCESS_WIFI_STATE,
            Permission.CHANGE_WIFI_STATE,
            Permission.ACCESS_FINE_LOCATION,
            Permission.ACCESS_COARSE_LOCATION,
            Permission.NEARBY_WIFI_DEVICES
        ])
        print("✓ WiFi permissions requested")
        return True
    except Exception as e:
        print(f"✗ Permission request failed: {e}")
        return False

class WiFiP2PManager:
    """WiFi Direct P2P Manager with crash protection"""
    
    def __init__(self):
        self.is_android = ANDROID_AVAILABLE
        self.manager = None
        self.channel = None
        self.service_info = None
        self.is_group_owner = False
        self.on_hazard_received = None
        self.initialized = False
        
        if self.is_android:
            Clock.schedule_once(lambda dt: self._init_android(), 0.5)
        else:
            print("⚠ Not on Android or modules unavailable - using demo mode")
    
    def _init_android(self):
        """Initialize Android WiFi P2P manager safely"""
        try:
            print("Initializing WiFi P2P Manager...")
            
            # Request permissions first
            if not request_wifi_permissions():
                print("✗ Could not request permissions")
                return
            
            # Get Android activity
            activity = PythonActivity.mActivity
            if not activity:
                print("✗ Could not get Android activity")
                return
            
            # Get WiFi P2P service
            wifi_service = activity.getSystemService(Context.WIFI_P2P_SERVICE)
            if not wifi_service:
                print("✗ WiFi P2P service not available on this device")
                return
            
            # Cast to WifiP2pManager
            self.manager = cast(WifiP2pManager, wifi_service)
            if not self.manager:
                print("✗ Failed to cast WiFi P2P Manager")
                return
            
            # Initialize channel
            self.channel = self.manager.initialize(
                activity,
                activity.getMainLooper(),
                None
            )
            
            if self.channel:
                self.initialized = True
                print("✓ WiFi P2P Manager initialized successfully")
            else:
                print("✗ Failed to initialize WiFi P2P channel")
                
        except Exception as e:
            print(f"✗ WiFi P2P initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
    
    def start_service_as_sender(self, hazard_data):
        """Broadcast hazard alert via WiFi Direct"""
        if not self.is_android or not self.initialized:
            print("📡 DEMO: Broadcasting hazard (WiFi P2P not available)")
            return
        
        try:
            print("Starting WiFi Direct broadcast...")
            
            # Create service metadata
            service_map = {
                "hazard_type": str(hazard_data.get("type", "unknown")),
                "latitude": str(hazard_data.get("latitude", 0.0)),
                "longitude": str(hazard_data.get("longitude", 0.0)),
                "severity": str(hazard_data.get("severity", "medium")),
                "timestamp": str(hazard_data.get("timestamp", 0))
            }
            
            # Create DNS-SD service
            self.service_info = WifiP2pDnsSdServiceInfo.newInstance(
                "_hazardalert",
                "_presence._tcp",
                service_map
            )
            
            # Define listener
            class AddServiceListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ WiFi Direct service registered")
                    self.parent.is_group_owner = True
                    self.parent._start_socket_server(self.hazard_data)
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    errors = {0: "ERROR", 1: "P2P_UNSUPPORTED", 2: "BUSY"}
                    print(f"✗ Service registration failed: {errors.get(reason, 'UNKNOWN')}")
            
            listener = AddServiceListener()
            listener.parent = self
            listener.hazard_data = hazard_data
            
            # Register service
            self.manager.addLocalService(self.channel, self.service_info, listener)
            print("✓ WiFi Direct broadcast started")
            
        except Exception as e:
            print(f"✗ Broadcast error: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_socket_server(self, hazard_data):
        """Start TCP server for data transmission"""
        def server_thread():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('0.0.0.0', 8888))
                server.listen(5)
                print("✓ Socket server listening on port 8888")
                
                while True:
                    client, addr = server.accept()
                    print(f"✓ Client connected: {addr}")
                    message = json.dumps(hazard_data).encode('utf-8')
                    client.send(message)
                    client.close()
            except Exception as e:
                print(f"✗ Socket server error: {e}")
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def start_discovery_as_receiver(self, on_hazard_callback):
        """Start scanning for WiFi Direct hazard alerts"""
        self.on_hazard_received = on_hazard_callback
        
        # Check if WiFi P2P is available
        if not self.is_android:
            print("⚠ Not on Android - simulating alert")
            Clock.schedule_once(lambda dt: self._simulate_alert(), 3)
            return
        
        if not self.initialized:
            print("⚠ WiFi P2P not initialized - simulating alert")
            Clock.schedule_once(lambda dt: self._simulate_alert(), 3)
            return
        
        # Start real WiFi P2P discovery
        print("Starting WiFi Direct discovery...")
        Clock.schedule_once(lambda dt: self._safe_start_discovery(), 0.5)
    
    def _safe_start_discovery(self, *args):
        """Safely start WiFi P2P discovery with error handling"""
        try:
            # Request permissions again
            request_wifi_permissions()
            
            # Define listeners with error handling
            class TxtListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdTxtRecordListener']
                
                @java_method('(Ljava/lang/String;Ljava/util/Map;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
                def onDnsSdTxtRecordAvailable(self, fullDomain, txtRecord, device):
                    try:
                        print(f"✓ Service discovered: {device.deviceName}")
                        hazard_metadata = {
                            "type": txtRecord.get("hazard_type"),
                            "latitude": float(txtRecord.get("latitude")),
                            "longitude": float(txtRecord.get("longitude")),
                            "severity": txtRecord.get("severity"),
                            "device": device
                        }
                        self.parent._handle_discovered_hazard(hazard_metadata)
                    except Exception as e:
                        print(f"✗ Error processing discovery: {e}")
            
            class ServiceListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdServiceResponseListener']
                
                @java_method('(Ljava/lang/String;Ljava/lang/String;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
                def onDnsSdServiceAvailable(self, instanceName, registrationType, device):
                    print(f"✓ Service: {instanceName}")
            
            txt_listener = TxtListener()
            txt_listener.parent = self
            service_listener = ServiceListener()
            
            # Set listeners
            self.manager.setDnsSdResponseListeners(
                self.channel,
                service_listener,
                txt_listener
            )
            
            # Create and add service request
            service_request = WifiP2pDnsSdServiceRequest.newInstance()
            
            class RequestListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ Service request added")
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Service request failed: {reason}")
            
            self.manager.addServiceRequest(
                self.channel,
                service_request,
                RequestListener()
            )
            
            # Start discovery
            class DiscoveryListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ WiFi Direct discovery started")
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Discovery failed: {reason}")
                    # Fallback to simulation
                    self.parent._simulate_alert()
            
            discovery_listener = DiscoveryListener()
            discovery_listener.parent = self
            
            self.manager.discoverServices(self.channel, discovery_listener)
            print("✓ Scanning for WiFi Direct alerts...")
            
        except Exception as e:
            print(f"✗ Discovery error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simulation
            self._simulate_alert()
    
    def _simulate_alert(self):
        """Simulate receiving an alert (fallback)"""
        print("⚠ Simulating hazard alert (WiFi P2P unavailable)")
        
        demo_hazard = {
            "type": "pothole",
            "latitude": 19.1246278,
            "longitude": 72.8373842,
            "severity": "high",
            "confidence": 0.92,
            "timestamp": 1699824000
        }
        
        if self.on_hazard_received:
            Clock.schedule_once(lambda dt: self.on_hazard_received(demo_hazard), 3)
    
    def _handle_discovered_hazard(self, hazard_metadata):
        """Handle discovered hazard"""
        try:
            print(f"Processing hazard: {hazard_metadata['type']}")
            # For demo, just simulate receiving the alert
            demo_hazard = {
                "type": hazard_metadata["type"],
                "latitude": hazard_metadata["latitude"],
                "longitude": hazard_metadata["longitude"],
                "severity": hazard_metadata["severity"],
                "confidence": 0.92,
                "timestamp": 1699824000
            }
            
            if self.on_hazard_received:
                Clock.schedule_once(lambda dt: self.on_hazard_received(demo_hazard), 0.5)
                
        except Exception as e:
            print(f"✗ Error handling hazard: {e}")
    
    def stop(self):
        """Stop WiFi P2P services"""
        if self.is_android and self.initialized and self.manager:
            try:
                if self.service_info:
                    self.manager.removeLocalService(self.channel, self.service_info, None)
                self.manager.stopPeerDiscovery(self.channel, None)
                print("✓ WiFi P2P stopped")
            except Exception as e:
                print(f"Stop error: {e}")



