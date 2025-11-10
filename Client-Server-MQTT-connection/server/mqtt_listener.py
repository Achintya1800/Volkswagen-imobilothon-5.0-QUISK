# # server/mqtt_listener.py
# import paho.mqtt.client as mqtt
# import json
# from shared.config import BROKER, BROKER_PORT, MQTT_TOPIC, SERVER_URL
# import requests

# CLIENT_ID = "server-mqtt-listener"

# def on_connect(client, userdata, flags, rc):
#     print("[MQTT SERVER LISTENER] connected with result code", rc)
#     client.subscribe(MQTT_TOPIC)

# def on_message(client, userdata, msg):
#     try:
#         payload = json.loads(msg.payload.decode())
#         print("[MQTT SERVER LISTENER] Received:", payload)
#         # Option A: Post to FastAPI endpoint so server dedup logic applies
#         resp = requests.post(f"{SERVER_URL}/upload_hazard", json=payload, timeout=5)
#         print("[MQTT SERVER LISTENER] Server response:", resp.json())
#     except Exception as e:
#         print("Error handling MQTT message:", e)

# if __name__ == "__main__":
#     mqttc = mqtt.Client(CLIENT_ID)
#     mqttc.on_connect = on_connect
#     mqttc.on_message = on_message
#     mqttc.connect(BROKER, BROKER_PORT, keepalive=60)
#     mqttc.loop_forever()


# server/mqtt_listener.py
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion  # Add this import
import json
from shared.config import BROKER, BROKER_PORT, MQTT_TOPIC, SERVER_URL
import requests

CLIENT_ID = "server-mqtt-listener"

def on_connect(client, userdata, flags, rc, properties=None):  # Add properties parameter
    print("[MQTT SERVER LISTENER] connected with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print("[MQTT SERVER LISTENER] Received:", payload)
        # Option A: Post to FastAPI endpoint so server dedup logic applies
        resp = requests.post(f"{SERVER_URL}/upload_hazard", json=payload, timeout=5)
        print("[MQTT SERVER LISTENER] Server response:", resp.json())
    except Exception as e:
        print("Error handling MQTT message:", e)

if __name__ == "__main__":
    mqttc = mqtt.Client(CallbackAPIVersion.VERSION2, CLIENT_ID)  # Updated this line
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.connect(BROKER, BROKER_PORT, keepalive=60)
    mqttc.loop_forever()