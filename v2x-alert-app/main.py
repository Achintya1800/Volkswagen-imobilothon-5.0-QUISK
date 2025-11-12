
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.utils import platform
import time

from wifi_p2p_manager import WiFiP2PManager
from hazard_service import HazardService
from utils.location import LocationManager

class V2XHazardAlertApp(App):
    def build(self):
        self.wifi_manager = WiFiP2PManager()
        self.hazard_service = HazardService()
        self.location_manager = LocationManager()
        
        # Start GPS
        self.location_manager.start(on_location=self.on_location_update)
        
        # Periodically update GPS status
        Clock.schedule_interval(self.check_gps_status, 2)  
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        layout.add_widget(Label(
            text='V2X Hazard Alert Demo',
            size_hint_y=0.1,
            font_size='24sp',
            bold=True
        ))
        
        # GPS Status
        self.gps_label = Label(
            text='GPS: Waiting...',
            size_hint_y=0.08,
            font_size='14sp'
        )
        layout.add_widget(self.gps_label)
        
        # Mode selection
        mode_layout = BoxLayout(size_hint_y=0.15, spacing=10)
        
        self.sender_btn = Button(
            text='Device A\n(Send Alert)',
            on_press=self.switch_to_sender
        )
        mode_layout.add_widget(self.sender_btn)
        
        self.receiver_btn = Button(
            text='Device B\n(Receive Alert)',
            on_press=self.switch_to_receiver
        )
        mode_layout.add_widget(self.receiver_btn)
        
        layout.add_widget(mode_layout)
        
        # Sender controls
        self.sender_panel = BoxLayout(orientation='vertical', spacing=5, size_hint_y=0.4)
        
        self.hazard_spinner = Spinner(
            text='Select Hazard Type',
            values=('pothole', 'speed_bump', 'debris', 'stalled_vehicle'),
            size_hint_y=None,
            height=44
        )
        self.sender_panel.add_widget(self.hazard_spinner)
        
        severity_layout = BoxLayout(size_hint_y=None, height=44, spacing=5)
        severity_layout.add_widget(Label(text='Severity:', size_hint_x=0.3))
        self.severity_spinner = Spinner(
            text='medium',
            values=('low', 'medium', 'high', 'critical'),
            size_hint_x=0.7
        )
        severity_layout.add_widget(self.severity_spinner)
        self.sender_panel.add_widget(severity_layout)
        
        self.broadcast_btn = Button(
            text='Broadcast Hazard Alert',
            on_press=self.broadcast_hazard,
            size_hint_y=None,
            height=60,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        self.sender_panel.add_widget(self.broadcast_btn)
        
        self.sender_panel.opacity = 0
        self.sender_panel.disabled = True
        layout.add_widget(self.sender_panel)
        
        # Receiver controls
        self.receiver_panel = BoxLayout(orientation='vertical', spacing=5, size_hint_y=0.2)
        
        self.scan_btn = Button(
            text='Start Scanning for Alerts',
            on_press=self.start_scanning,
            size_hint_y=None,
            height=60,
            background_color=(0.2, 0.6, 0.2, 1)
        )
        self.receiver_panel.add_widget(self.scan_btn)
        
        self.receiver_panel.opacity = 0
        self.receiver_panel.disabled = True
        layout.add_widget(self.receiver_panel)
        
        # Status/Alert display
        self.status_label = Label(
            text='Select mode to begin',
            size_hint_y=0.27,
            font_size='16sp',
            markup=True,
            halign='center',
            valign='middle'
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        layout.add_widget(self.status_label)
        
        return layout
    
    # def on_location_update(self, lat, lon):
    #     """Called when GPS location updates"""
    #     self.gps_label.text = f'GPS: {lat:.6f}, {lon:.6f}'
    def on_location_update(self, lat, lon):
        if lat is not None and lon is not None:
            self.gps_label.text = f'GPS: {lat:.6f}, {lon:.6f} ✓'
            print(f"GPS UPDATED: {lat}, {lon}")
        else:
            self.gps_label.text = 'GPS: No signal'


    
    def switch_to_sender(self, instance):
        """Switch to sender mode"""
        self.sender_panel.opacity = 1
        self.sender_panel.disabled = False
        self.receiver_panel.opacity = 0
        self.receiver_panel.disabled = True
        
        self.sender_btn.background_color = (0.2, 0.6, 0.2, 1)
        self.receiver_btn.background_color = (0.5, 0.5, 0.5, 1)
        
        self.status_label.text = '[b]SENDER MODE[/b]\nSelect hazard and broadcast'
    
    def switch_to_receiver(self, instance):
        """Switch to receiver mode"""
        self.receiver_panel.opacity = 1
        self.receiver_panel.disabled = False
        self.sender_panel.opacity = 0
        self.sender_panel.disabled = True
        
        self.receiver_btn.background_color = (0.2, 0.6, 0.2, 1)
        self.sender_btn.background_color = (0.5, 0.5, 0.5, 1)
        
        self.status_label.text = '[b]RECEIVER MODE[/b]\nStart scanning to detect alerts'
    
    def broadcast_hazard(self, instance):
        """Broadcast hazard alert via WiFi P2P"""
        lat, lon = self.location_manager.get_location()
        
        if lat is None or lon is None:
            self.status_label.text = '[color=ff0000]Error: GPS not ready[/color]'
            return
        
        hazard_type = self.hazard_spinner.text
        if hazard_type == 'Select Hazard Type':
            self.status_label.text = '[color=ff0000]Please select hazard type[/color]'
            return
        
        # Create hazard alert
        hazard = self.hazard_service.create_hazard_alert(
            hazard_type=hazard_type,
            latitude=lat,
            longitude=lon,
            severity=self.severity_spinner.text,
            confidence=0.92,
            size={"width": 0.5, "depth": 0.15},
            on_road=True
        )
        
        self.status_label.text = f'[b][color=ff8800]BROADCASTING...[/color][/b]\n\n' \
                                 f'{self.hazard_service.format_alert_message(hazard)}\n\n' \
                                 f'Location: {lat:.6f}, {lon:.6f}\n' \
                                 f'Geofence: 500m radius'
        
        # Start WiFi P2P broadcast
        self.wifi_manager.start_service_as_sender(hazard)
        
        print(f"Broadcasting hazard: {hazard}")

    def check_gps_status(self, dt):
        """Periodically check and display GPS status"""
        lat, lon = self.location_manager.get_location()
        if lat is None or lon is None:
        # GPS not ready - show helpful message
            self.gps_label.text = 'GPS: Acquiring signal... (go outside)'
        else:
        # GPS ready - update label
            self.gps_label.text = f'GPS: {lat:.6f}, {lon:.6f} ✓'

    def start_scanning(self, instance):
        """Start scanning for hazard alerts"""
        lat, lon = self.location_manager.get_location()
        
        if lat is None or lon is None:
            self.status_label.text = '[color=ff0000]Error: GPS not ready[/color]'
            return
        
        self.status_label.text = '[b][color=00ff00]SCANNING...[/color][/b]\n\n' \
                                 f'Current Location:\n{lat:.6f}, {lon:.6f}\n\n' \
                                 'Waiting for hazard alerts...'
        
        # Start WiFi P2P discovery
        self.wifi_manager.start_discovery_as_receiver(
            on_hazard_callback=self.on_hazard_received
        )
        
        self.scan_btn.text = 'Scanning... (Tap to stop)'
        self.scan_btn.on_press = self.stop_scanning
    
    def stop_scanning(self, instance):
        """Stop scanning"""
        self.wifi_manager.stop()
        self.scan_btn.text = 'Start Scanning for Alerts'
        self.scan_btn.on_press = self.start_scanning
        self.status_label.text = 'Scanning stopped'
    
    def on_hazard_received(self, hazard):
        """Called when hazard alert is received"""
        Clock.schedule_once(lambda dt: self._display_hazard_alert(hazard))
    
    def _display_hazard_alert(self, hazard):
        """Display received hazard alert"""
        from geofence import haversine_distance
        
        lat, lon = self.location_manager.get_location()
        distance = haversine_distance(
            lat, lon,
            hazard['latitude'],
            hazard['longitude']
        )
        
        self.status_label.text = f'[b][color=ff0000]⚠ HAZARD ALERT! ⚠[/color][/b]\n\n' \
                                 f'{self.hazard_service.format_alert_message(hazard)}\n\n' \
                                 f'Distance: {distance:.0f}m ahead\n' \
                                 f'Location: {hazard["latitude"]:.6f}, {hazard["longitude"]:.6f}'
        
        print(f"ALERT DISPLAYED: {hazard}")
    
    def on_stop(self):
        """Cleanup on app close"""
        self.wifi_manager.stop()
        self.location_manager.stop()

if __name__ == '__main__':
    V2XHazardAlertApp().run()




