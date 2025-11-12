# client/mqtt_client.py
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion  # Add this import
import json
import random
import threading
from shared.config import BROKER, BROKER_PORT, MQTT_TOPIC

class MQTTClient:
    def __init__(self, client_id=None, on_message_callback=None):
        self.client_id = client_id or f"driver-{random.randint(1000,9999)}"
        self.mqtt = mqtt.Client(CallbackAPIVersion.VERSION2, self.client_id)  # Updated this line
        if on_message_callback:
            self.mqtt.on_message = on_message_callback
        self.mqtt.on_connect = self._on_connect
        self.mqtt_lock = threading.Lock()

    def _on_connect(self, client, userdata, flags, rc, properties=None):  # Add properties parameter
        print(f"[{self.client_id}] MQTT connected (rc={rc}). Subscribing to topic.")
        client.subscribe(MQTT_TOPIC)

    def start(self):
        self.mqtt.connect(BROKER, BROKER_PORT, keepalive=60)
        thread = threading.Thread(target=self.mqtt.loop_forever, daemon=True)
        thread.start()

    def publish_hazard(self, data):
        with self.mqtt_lock:
            self.mqtt.publish(MQTT_TOPIC, json.dumps(data))