"""U by Moen cloud client.

Taken from AshleyF/sandbox `moen/moen_api.py`. That file on main still has
conflict markers; this copy keeps the certifi trust-store fix from that work.
Auth and status go to https://www.moen-iot.com. On, off, preset, and
temperature go out as Pusher `client-state-desired` events, which is what
the Moen app sends.
"""

import json
import logging
import os
import time
import threading

import certifi
import pysher
import requests

logging.getLogger("pysher").setLevel(logging.CRITICAL)
logging.getLogger("pysher.connection").setLevel(logging.CRITICAL)
logging.getLogger("websocket").setLevel(logging.CRITICAL)

PUSHER_KEY = "dcc28ccb5296f18f8eae"
PUSHER_CLUSTER = "us2"


class MoenClient:
    BASE_URL = "https://www.moen-iot.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self.token = None
        self._pusher = None
        self._channel_name = None
        self._connected = threading.Event()
        self._subscribed = threading.Event()

    def _require_auth(self):
        if not self.token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        self.session.headers["User-Token"] = self.token

    def authenticate(self, email, password):
        resp = self.session.get(
            f"{self.BASE_URL}/v2/authenticate",
            params={"email": email, "password": password},
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data.get("token")
        return data

    def get_showers(self):
        self._require_auth()
        resp = self.session.get(f"{self.BASE_URL}/v2/showers")
        resp.raise_for_status()
        return resp.json()

    def get_shower_details(self, serial):
        self._require_auth()
        resp = self.session.get(f"{self.BASE_URL}/v5/showers/{serial}")
        resp.raise_for_status()
        return resp.json()

    def _ensure_pusher(self, serial):
        if self._pusher and self._subscribed.is_set():
            return
        # websocket-client uses the interpreter CA store, which can be incomplete
        # on a Mac. Requests already depends on certifi, so use that bundle.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        details = self.get_shower_details(serial)
        self._channel_name = f"private-{details['channel']}"
        self.session.headers["Shower-Token"] = details["token"]
        self.session.post(f"{self.BASE_URL}/v2/capabilities?name=mobile_supports_pusher")
        self.session.headers.pop("Shower-Token", None)
        self._connected.clear()
        self._subscribed.clear()
        self._pusher = pysher.Pusher(
            key=PUSHER_KEY,
            cluster=PUSHER_CLUSTER,
            auth_endpoint=f"{self.BASE_URL}/v3/pusher-auth",
            auth_endpoint_headers={"User-Token": self.token},
            log_level=logging.CRITICAL,
        )

        def on_connect(_data):
            channel = self._pusher.subscribe(self._channel_name)
            channel.bind("pusher_internal:subscription_succeeded", lambda _event: self._subscribed.set())
            self._connected.set()

        self._pusher.connection.bind("pusher:connection_established", on_connect)
        self._pusher.connect()
        if not self._connected.wait(timeout=10):
            raise RuntimeError("Failed to connect to Pusher")
        if not self._subscribed.wait(timeout=10):
            raise RuntimeError("Failed to subscribe to shower channel")

    def _send_control(self, serial, action, params=None):
        self._ensure_pusher(serial)
        data = {"type": "control", "data": {"action": action, "params": params or {}}}
        msg = json.dumps({
            "event": "client-state-desired",
            "channel": self._channel_name,
            "data": json.dumps(data),
        })
        self._pusher.connection.socket.send(msg)

    def disconnect(self):
        if self._pusher:
            try:
                self._pusher.disconnect()
            except Exception:
                pass
            self._pusher = None
            self._connected.clear()
            self._subscribed.clear()

    def turn_on(self, serial, preset=0):
        """Turn the shower on. Preset 0 is whatever it used last."""
        self._send_control(serial, "shower_on", {"preset": str(preset)})

    def turn_off(self, serial):
        self._send_control(serial, "shower_off")

    def set_temperature(self, serial, temp_f):
        self._send_control(serial, "temperature_set", {"target_temperature": temp_f})

    def activate_preset(self, serial, preset_index):
        details = self.get_shower_details(serial)
        preset = next((item for item in details.get("presets", []) if item.get("position") == preset_index), None)
        if not preset:
            raise ValueError(f"Preset {preset_index} not found")
        config = {
            "active_preset": preset_index,
            "title": preset.get("title", ""),
            "greeting": preset.get("greeting", ""),
            "target_temperature": preset.get("target_temperature", 100),
            "ready_pauses_water": preset.get("ready_pauses_water", False),
            "ready_pushes_notification": preset.get("ready_pushes_notification", False),
            "ready_sounds_alert": preset.get("ready_sounds_alert", False),
            "timer_enabled": preset.get("timer_enabled", False),
            "timer_length": preset.get("timer_length", 600),
            "timer_ends_shower": preset.get("timer_ends_shower", False),
            "timer_sounds_alert": preset.get("timer_sounds_alert", True),
            "outlets": preset.get("outlets", []),
        }
        self._send_control(serial, "shower_set", config)
        time.sleep(0.5)
        self._send_control(serial, "shower_on", {"preset": str(preset_index)})
