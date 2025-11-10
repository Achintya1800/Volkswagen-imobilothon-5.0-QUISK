# client/main_client.py
import time
import random
import threading
import json
import os
import requests
from shared.config import SERVER_URL
from client.mqtt_client import MQTTClient
from math import radians, sin, cos, sqrt, atan2

CACHE_FILE = "client/hazard_cache.json"
os.makedirs("client", exist_ok=True)

def is_nearby(lat1, lon1, lat2, lon2, radius_m=100):
    # Haversine
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c < radius_m

# Simple on_message handler to show alerts and geofence
# def on_message(client, userdata, msg):
#     payload = json.loads(msg.payload.decode())
#     # Simulate current GPS of this client
#     my_lat, my_lon = userdata.get("gps", (19.0760, 72.8777))
#     if is_nearby(my_lat, my_lon, payload['latitude'], payload['longitude'], radius_m=150):
#         print(">>> LOCAL ALERT - hazard is nearby!", payload)
#     else:
#         print("... received hazard (not nearby):", payload)
def on_message(client, userdata, msg, properties=None):  # Add properties parameter
    payload = json.loads(msg.payload.decode())
    # Simulate current GPS of this client
    my_lat, my_lon = userdata.get("gps", (19.0760, 72.8777))
    if is_nearby(my_lat, my_lon, payload['latitude'], payload['longitude'], radius_m=150):
        print(">>> LOCAL ALERT - hazard is nearby!", payload)
    else:
        print("... received hazard (not nearby):", payload)
class ClientApp:
    def __init__(self):
        self.gps = (19.0760 + random.random()/100, 72.8777 + random.random()/100)
        self.mqtt_client = MQTTClient(on_message_callback=self._wrapped_on_message)
        # store gps in userdata for callback
        self.mqtt_client.mqtt.user_data_set({"gps": self.gps})
        self.online = True
        self.lock = threading.Lock()
        self.mqtt_client.start()

    def _wrapped_on_message(self, client, userdata, msg):
        on_message(client, userdata, msg)

    def cache_hazard(self, payload):
        with self.lock:
            cache = []
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r") as f:
                    cache = json.load(f)
            cache.append(payload)
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            print("[CLIENT] Cached hazard locally (offline).")

    def flush_cache(self):
        if not os.path.exists(CACHE_FILE):
            return
        with self.lock:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            for p in cache:
                try:
                    resp = requests.post(f"{SERVER_URL}/upload_hazard", json=p, timeout=5)
                    print("[CLIENT] Flushed cached hazard -> server:", resp.json())
                except Exception as e:
                    print("[CLIENT] Failed to flush cached hazard:", e)
                    return
            os.remove(CACHE_FILE)

    def send_to_server(self, payload):
        try:
            resp = requests.post(f"{SERVER_URL}/upload_hazard", json=payload, timeout=5)
            print("[CLIENT] Server response:", resp.json())
            return True
        except Exception as e:
            print("[CLIENT] Error sending to server, caching:", e)
            return False

    def simulate_detection_loop(self):
        while True:
            # Simulate frame capture & model outputs
            lat = self.gps[0] + random.uniform(-0.0005, 0.0005)
            lon = self.gps[1] + random.uniform(-0.0005, 0.0005)
            hazard_type = random.choice(["pothole", "bump", "stalled_vehicle"])
            # simulate a fusion score from models
            fusion_score = random.random()  # 0..1

            payload = {
                "latitude": lat,
                "longitude": lon,
                "confidence": round(fusion_score, 3),
                "hazard_type": hazard_type
            }

            print(f"[CLIENT] Detected: {payload}")

            if fusion_score >= 0.85:
                # high confidence -> publish to MQTT and REST
                self.mqtt_client.publish_hazard(payload)
                sent = self.send_to_server(payload)
                if not sent:
                    self.cache_hazard(payload)

            elif 0.6 <= fusion_score < 0.85:
                # medium -> ask driver via GEN-AI verification (here simulate auto-yes 70% of time)
                user_confirm = random.random() < 0.7
                print("[CLIENT] Medium confidence. Driver verification:", user_confirm)
                if user_confirm:
                    self.mqtt_client.publish_hazard(payload)
                    sent = self.send_to_server(payload)
                    if not sent:
                        self.cache_hazard(payload)
            else:
                # low -> ignore
                print("[CLIENT] Low confidence, ignored.")

            # try flushing cache periodically
            self.flush_cache()
            time.sleep(4 + random.random()*4)  # wait 4-8 seconds

if __name__ == "__main__":
    app = ClientApp()
    try:
        app.simulate_detection_loop()
    except KeyboardInterrupt:
        print("Client stopped.")
