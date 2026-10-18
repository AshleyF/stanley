"""Kasa lights listed in home.json."""

def clamp(value, lo, hi, default):
    if value is None:
        return default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))

def find_named(items, name):
    wanted = (name or "").casefold().strip()
    for item in items:
        if item["name"].casefold() == wanted:
            return item
    hits = [item for item in items if wanted and wanted in item["name"].casefold()]
    return hits[0] if len(hits) == 1 else None

def listing(items):
    return "\n".join(f"- {item['name']}: {item.get('description') or item.get('host')}" for item in items)

class Lights:
    name = "set_light"

    def handles(self, name):
        return name == self.name

    def instructions(self, home):
        lights = home["lights"]
        if not lights:
            return ""
        return (
            "You can change these lights:\n"
            f"{listing(lights)}\n"
            "Call set_light only when they clearly ask to turn one of these on or off or to change its color. "
            "Use that light's name. "
            "Choose hue from 0 to 360, saturation from 0 to 100, and brightness from 0 to 100."
        )

    def schemas(self, home):
        lights = home["lights"]
        if not lights:
            return []
        return [{
            "type": "function",
            "name": self.name,
            "description": "Turn one listed light on or off, and set its color as HSV. A color turns it on.\n" + listing(lights),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [light["name"] for light in lights]},
                    "on": {"type": "boolean"},
                    "hue": {"type": "integer", "minimum": 0, "maximum": 360},
                    "saturation": {"type": "integer", "minimum": 0, "maximum": 100},
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["name", "on"],
            },
        }]

    async def call(self, _name, args, home):
        from kasa import Discover, Module
        import contextlib
        light = find_named(home["lights"], args.get("name"))
        if light is None:
            print(f"\nno light named {args.get('name')}", flush=True)
            return {"error": f"no light named {args.get('name')}"}
        device = None
        try:
            device = await Discover.discover_single(light["host"], timeout=5)
            await device.update()
            label = light["name"]
            if not bool(args.get("on")):
                await device.turn_off()
                print(f"\n{label} off", flush=True)
                return {"name": label, "on": False}
            await device.turn_on()
            if args.get("hue") is None:
                print(f"\n{label} on", flush=True)
                return {"name": label, "on": True}
            bulb = device.modules.get(Module.Light)
            if bulb is None:
                print(f"\n{label} has no color", flush=True)
                return {"name": label, "on": True, "error": "no color"}
            hsv = (
                clamp(args.get("hue"), 0, 360, 0),
                clamp(args.get("saturation"), 0, 100, 100),
                clamp(args.get("brightness"), 0, 100, 100),
            )
            await bulb.set_hsv(*hsv)
            print(f"\n{label} on, hsv {hsv[0]} {hsv[1]} {hsv[2]}", flush=True)
            return {"name": label, "on": True, "hue": hsv[0], "saturation": hsv[1], "brightness": hsv[2]}
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            print(f"\n{text}", flush=True)
            return {"error": text}
        finally:
            if device is not None:
                with contextlib.suppress(Exception):
                    await device.disconnect()
