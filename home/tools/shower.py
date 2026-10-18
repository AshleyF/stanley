"""U by Moen shower listed in home.json.

Each call logs in, sends one command, and disconnects. Idle listening
does not hold a Moen or Pusher connection.
"""

import asyncio
import os

from .lights import find_named

class Shower:
    name = "set_shower"

    def handles(self, name):
        return name == self.name

    def instructions(self, home):
        showers = home["showers"]
        if not showers:
            return ""
        return (
            "You can control these showers. Temperatures are Fahrenheit.\n"
            f"{listing(showers)}\n"
            "Call set_shower only when they clearly ask to turn a shower on or off, "
            "start one of its presets, or change its temperature. Use its name. "
            "Never ask for a temperature above that shower's maximum. "
            "If the result temperature is lower than they asked, say the number that was set."
        )

    def schemas(self, home):
        showers = home["showers"]
        if not showers:
            return []
        presets = sorted({preset["name"] for shower in showers for preset in shower["presets"]})
        preset = {"type": "string"}
        if presets:
            preset["enum"] = presets
        return [{
            "type": "function",
            "name": self.name,
            "description": "Turn one listed shower on or off, start a preset, or set its temperature in Fahrenheit.\n" + listing(showers),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [shower["name"] for shower in showers]},
                    "action": {"type": "string", "enum": ["on", "off", "temperature"]},
                    "preset": preset,
                    "temperature": {"type": "integer"},
                },
                "required": ["name", "action"],
            },
        }]

    async def call(self, _name, args, home):
        return await asyncio.to_thread(self.run, args, home)

    def run(self, args, home):
        shower = find_named(home["showers"], args.get("name"))
        action = str(args.get("action") or "").casefold()
        if shower is None:
            print(f"\nno shower named {args.get('name')}", flush=True)
            return {"error": f"no shower named {args.get('name')}"}
        if action not in ("on", "off", "temperature"):
            return {"error": "action must be on, off, or temperature"}
        email, password = os.environ.get("MOEN_EMAIL"), os.environ.get("MOEN_PASSWORD")
        if not email or not password:
            print("\nMoen login is not set", flush=True)
            return {"error": "Set MOEN_EMAIL and MOEN_PASSWORD in .env"}
        temp = None
        if action != "off" and (action == "temperature" or args.get("temperature") is not None):
            temp = bounded_temperature(args.get("temperature"), shower["max_temperature"])
            if temp is None:
                return {"error": "temperature must be a number in Fahrenheit"}
        preset = find_preset(shower["presets"], args.get("preset")) if args.get("preset") else None
        if args.get("preset") and preset is None:
            names = ", ".join(item["name"] for item in shower["presets"]) or "none"
            return {"error": f"no preset named {args.get('preset')} (have {names})"}
        client = None
        try:
            from .moen import MoenClient
            client = MoenClient()
            client.authenticate(email, password)
            serial = shower["serial"] or only_serial(client)
            if action == "off":
                client.turn_off(serial)
                print(f"\n{shower['name']} off", flush=True)
                return {"name": shower["name"], "action": "off"}
            result = {"name": shower["name"], "action": "on" if action == "on" else "temperature"}
            if action == "on":
                if preset:
                    client.activate_preset(serial, preset["index"])
                    result["preset"] = preset["name"]
                else:
                    client.turn_on(serial, preset=0)
            if temp is not None:
                client.set_temperature(serial, temp)
                result["temperature"] = temp
                asked = args.get("temperature")
                try:
                    asked_number = int(round(float(asked)))
                except (TypeError, ValueError):
                    asked_number = None
                if asked_number is not None and temp != asked_number:
                    result["asked"] = asked_number
            print(f"\n{shower['name']} {result.get('preset') or action}" + (f" {result['temperature']}F" if "temperature" in result else ""), flush=True)
            return result
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            print(f"\n{text}", flush=True)
            return {"error": text}
        finally:
            if client is not None:
                client.disconnect()

def listing(showers):
    lines = []
    for shower in showers:
        presets = ", ".join(preset["name"] for preset in shower["presets"]) or "last used"
        text = shower.get("description") or "shower"
        lines.append(f"- {shower['name']}: {text} Presets: {presets}. Maximum {shower['max_temperature']}°F.")
    return "\n".join(lines)

def bounded_temperature(value, maximum):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(70, min(maximum, number))

def find_preset(presets, name):
    found = find_named(presets, name)
    if found:
        return found
    text = str(name or "").strip()
    if text.isdigit():
        index = int(text)
        for preset in presets:
            if preset["index"] == index:
                return preset
    return None

def only_serial(client):
    showers = client.get_showers() or []
    if len(showers) == 1:
        serial = showers[0].get("serial_number") or showers[0].get("serial")
        if serial:
            return serial
    names = ", ".join(item.get("name") or item.get("serial_number") or "unnamed" for item in showers) or "none"
    raise RuntimeError(f"set serial on the shower in home.json (account has {names})")
