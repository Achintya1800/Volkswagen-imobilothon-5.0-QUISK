


"""WiFi Direct P2P Manager for hazard alert transmission"""
import json
import socket
import threading
from kivy.utils import platform

# Import Android modules (only imported when running on Android device)
if platform == 'android':
    from jnius import autoclass, cast, PythonJavaClass, java_method
    from android.permissions import request_permissions, Permission
    
    # Java classes
    WifiP2pManager = autoclass('android.net.wifi.p2p.WifiP2pManager')
    WifiP2pConfig = autoclass('android.net.wifi.p2p.WifiP2pConfig')
    WifiP2pDnsSdServiceInfo = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceInfo')
    WifiP2pDnsSdServiceRequest = autoclass('android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceRequest')
    Context = autoclass('android.content.Context')
    PythonActivity = autoclass('org.kivy.android.PythonActivity')

def request_wifi_permissions():
    """Request WiFi P2P permissions - REAL Android implementation"""
    if platform != 'android':
        print("Desktop mode - WiFi P2P not available")
        return
        
    request_permissions([
        Permission.ACCESS_WIFI_STATE,
        Permission.CHANGE_WIFI_STATE,
        Permission.ACCESS_FINE_LOCATION,
        Permission.NEARBY_WIFI_DEVICES
    ])

class WiFiP2PManager:
    """Manages WiFi Direct P2P connections for hazard alerts"""
    
    def __init__(self):
        self.is_android = platform == 'android'
        self.manager = None
        self.channel = None
        self.receiver = None
        self.service_info = None
        self.is_group_owner = False
        self.on_hazard_received = None
        
        if self.is_android:
            self._init_android()
        else:
            print("WARNING: Not on Android - WiFi P2P disabled")
    
    def _init_android(self):
        """Initialize Android WiFi P2P manager - REAL Android code"""
        request_wifi_permissions()
        
        try:
            activity = PythonActivity.mActivity
            self.manager = cast(
                WifiP2pManager,
                activity.getSystemService(Context.WIFI_P2P_SERVICE)
            )
            self.channel = self.manager.initialize(
                activity,
                activity.getMainLooper(),
                None
            )
            print("✓ WiFi P2P Manager initialized successfully")
        except Exception as e:
            print(f"✗ WiFi P2P init error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_service_as_sender(self, hazard_data):
        """
        Start WiFi Direct service as hazard alert sender (Group Owner)
        REAL Android WiFi Direct implementation - NO MOCK
        
        Args:
            hazard_data: dict containing hazard information
        """
        if not self.is_android:
            print("Desktop mode - Cannot broadcast WiFi P2P")
            return
        
        if not self.manager:
            print("✗ WiFi P2P Manager not initialized")
            return
        
        try:
            # Create service info with hazard metadata
            service_map = {
                "hazard_type": str(hazard_data.get("type", "unknown")),
                "latitude": str(hazard_data.get("latitude", 0.0)),
                "longitude": str(hazard_data.get("longitude", 0.0)),
                "severity": str(hazard_data.get("severity", "medium")),
                "timestamp": str(hazard_data.get("timestamp", 0))
            }
            
            print(f"Creating WiFi Direct service with data: {service_map}")
            
            # Create DNS-SD service
            self.service_info = WifiP2pDnsSdServiceInfo.newInstance(
                "_hazardalert",      # Instance name
                "_presence._tcp",     # Service type
                service_map          # TXT record map
            )
            
            # Add local service
            class AddServiceListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ WiFi Direct service registered successfully")
                    self.parent.is_group_owner = True
                    # Start socket server
                    self.parent._start_socket_server(hazard_data)
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Service registration failed - Error code: {reason}")
                    error_msgs = {
                        0: "ERROR",
                        1: "P2P_UNSUPPORTED",
                        2: "BUSY"
                    }
                    print(f"  Reason: {error_msgs.get(reason, 'UNKNOWN')}")
            
            listener = AddServiceListener()
            listener.parent = self
            listener.hazard_data = hazard_data
            
            self.manager.addLocalService(
                self.channel,
                self.service_info,
                listener
            )
            
            print("📡 Broadcasting hazard alert via WiFi Direct...")
            
        except Exception as e:
            print(f"✗ Error starting sender service: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_socket_server(self, hazard_data):
        """Start TCP socket server to send full hazard data - REAL implementation"""
        def server_thread():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # Group Owner uses 192.168.49.1
                server.bind(('0.0.0.0', 8888))
                server.listen(5)
                
                print("✓ Socket server listening on port 8888 (Group Owner)")
                
                while True:
                    client, addr = server.accept()
                    print(f"✓ Client connected from {addr}")
                    
                    # Send hazard JSON
                    message = json.dumps(hazard_data).encode('utf-8')
                    client.send(message)
                    print(f"✓ Sent hazard data to {addr}")
                    
                    client.close()
                    
            except Exception as e:
                print(f"✗ Socket server error: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def start_discovery_as_receiver(self, on_hazard_callback):
        """
        Start WiFi Direct service discovery as receiver
        REAL Android WiFi Direct implementation - NO MOCK
        
        Args:
            on_hazard_callback: Function called when hazard is discovered
        """
        if not self.is_android:
            print("Desktop mode - Cannot discover WiFi P2P services")
            return
        
        if not self.manager:
            print("✗ WiFi P2P Manager not initialized")
            return
        
        self.on_hazard_received = on_hazard_callback
        
        try:
            # DNS-SD TXT record listener
            class TxtRecordListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdTxtRecordListener']
                
                @java_method('(Ljava/lang/String;Ljava/util/Map;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
                def onDnsSdTxtRecordAvailable(self, fullDomain, txtRecord, device):
                    print(f"✓ WiFi Direct service discovered: {fullDomain}")
                    print(f"  Device: {device.deviceName}")
                    
                    try:
                        # Extract hazard metadata
                        hazard_metadata = {
                            "type": txtRecord.get("hazard_type"),
                            "latitude": float(txtRecord.get("latitude")),
                            "longitude": float(txtRecord.get("longitude")),
                            "severity": txtRecord.get("severity"),
                            "device": device
                        }
                        
                        print(f"  Hazard: {hazard_metadata['type']} at {hazard_metadata['latitude']}, {hazard_metadata['longitude']}")
                        
                        # Check geofence and connect
                        self.parent._handle_discovered_hazard(hazard_metadata)
                    except Exception as e:
                        print(f"✗ Error processing discovered service: {e}")
            
            # DNS-SD service listener
            class ServiceListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$DnsSdServiceResponseListener']
                
                @java_method('(Ljava/lang/String;Ljava/lang/String;Landroid/net/wifi/p2p/WifiP2pDevice;)V')
                def onDnsSdServiceAvailable(self, instanceName, registrationType, device):
                    print(f"✓ Service available: {instanceName} ({registrationType})")
                    print(f"  Device: {device.deviceName} [{device.deviceAddress}]")
            
            txt_listener = TxtRecordListener()
            txt_listener.parent = self
            
            service_listener = ServiceListener()
            
            # Set response listeners
            self.manager.setDnsSdResponseListeners(
                self.channel,
                service_listener,
                txt_listener
            )
            
            # Create service request
            service_request = WifiP2pDnsSdServiceRequest.newInstance()
            
            # Add service request
            class RequestListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ Service request added successfully")
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Service request failed - Error code: {reason}")
            
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
                    print("✓ WiFi Direct service discovery started successfully")
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Discovery start failed - Error code: {reason}")
            
            self.manager.discoverServices(
                self.channel,
                DiscoveryListener()
            )
            
            print("📡 Scanning for WiFi Direct hazard alerts...")
            
        except Exception as e:
            print(f"✗ Error starting discovery: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_discovered_hazard(self, hazard_metadata):
        """Handle discovered hazard - check geofence and connect"""
        from geofence import is_within_geofence
        from utils.location import LocationManager
        
        # Get current location
        loc_mgr = LocationManager()
        current_lat, current_lon = loc_mgr.get_location()
        
        if current_lat is None or current_lon is None:
            print("⚠ Warning: GPS not ready - cannot check geofence")
            print("  Connecting anyway for demo purposes")
            # For demo, connect even without GPS
            self._connect_to_sender(hazard_metadata["device"])
            return
        
        # Check geofence
        within_fence, distance = is_within_geofence(
            current_lat, current_lon,
            hazard_metadata["latitude"],
            hazard_metadata["longitude"],
            radius_m=500  # 500 meter geofence
        )
        
        print(f"Distance to hazard: {distance:.1f}m")
        
        if within_fence:
            print(f"⚠ ENTERING GEOFENCE - Connecting to receive alert")
            self._connect_to_sender(hazard_metadata["device"])
        else:
            print(f"Outside geofence ({distance:.1f}m > 500m) - NOT connecting")
    
    def _connect_to_sender(self, device):
        """Connect to sender device to receive full hazard data"""
        try:
            config = WifiP2pConfig()
            config.deviceAddress = device.deviceAddress
            
            class ConnectListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ActionListener']
                
                @java_method('()V')
                def onSuccess(self):
                    print("✓ Connected to sender device")
                    # Get group info and connect via socket
                    self.parent._get_connection_info()
                
                @java_method('(I)V')
                def onFailure(self, reason):
                    print(f"✗ Connection failed - Error code: {reason}")
            
            listener = ConnectListener()
            listener.parent = self
            
            self.manager.connect(self.channel, config, listener)
            print(f"Connecting to device: {device.deviceName}...")
            
        except Exception as e:
            print(f"✗ Connection error: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_connection_info(self):
        """Get connection info and receive hazard data"""
        try:
            class ConnectionInfoListener(PythonJavaClass):
                __javainterfaces__ = ['android.net.wifi.p2p.WifiP2pManager$ConnectionInfoListener']
                
                @java_method('(Landroid/net/wifi/p2p/WifiP2pInfo;)V')
                def onConnectionInfoAvailable(self, info):
                    if info.groupFormed:
                        go_address = info.groupOwnerAddress.getHostAddress()
                        print(f"✓ Group formed - Group Owner IP: {go_address}")
                        # Receive hazard data
                        self.parent._receive_hazard_data(go_address)
                    else:
                        print("⚠ Group not formed yet")
            
            listener = ConnectionInfoListener()
            listener.parent = self
            
            self.manager.requestConnectionInfo(self.channel, listener)
            
        except Exception as e:
            print(f"✗ Connection info error: {e}")
            import traceback
            traceback.print_exc()
    
    def _receive_hazard_data(self, go_address):
        """Receive full hazard data via TCP socket"""
        def receive_thread():
            try:
                print(f"Connecting to socket server at {go_address}:8888...")
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(10)  # 10 second timeout
                client.connect((go_address, 8888))
                
                print("✓ Connected to socket server")
                data = client.recv(4096).decode('utf-8')
                hazard = json.loads(data)
                
                print(f"✓ Received full hazard data: {hazard}")
                
                # Trigger callback
                if self.on_hazard_received:
                    self.on_hazard_received(hazard)
                
                client.close()
                
            except socket.timeout:
                print("✗ Socket connection timeout")
            except Exception as e:
                print(f"✗ Receive error: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=receive_thread, daemon=True).start()
    
    def stop(self):
        """Stop WiFi P2P services"""
        if not self.is_android:
            return
        
        try:
            if self.service_info and self.manager:
                self.manager.removeLocalService(self.channel, self.service_info, None)
            if self.manager:
                self.manager.stopPeerDiscovery(self.channel, None)
            print("✓ WiFi P2P stopped")
        except Exception as e:
            print(f"WiFi P2P stop error: {e}")




