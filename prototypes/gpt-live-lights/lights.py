"""One realtime session for the reading lamp at 192.168.4.38."""

import asyncio, base64, contextlib, json, os, signal, sys, threading
from pathlib import Path

HOST, RATE = "192.168.4.38", 24000
SESSION = {
    "type": "realtime",
    "model": "gpt-realtime-2.1",
    "output_modalities": ["audio"],
    "instructions": (
        "You control only the living-room reading lamp. "
        "Call set_lamp only when the person clearly asks to turn that lamp on or off or to change its color. "
        "Choose hue from 0 to 360, saturation from 0 to 100, and brightness from 0 to 100. "
        "If the audio is silence, noise, or about anything else, call wait_for_user. "
        "Say nothing unless you changed the lamp."
    ),
    "tool_choice": "auto",
    "audio": {
        "input": {"format": {"type": "audio/pcm", "rate": RATE}},
        "output": {"format": {"type": "audio/pcm", "rate": RATE}, "voice": "marin"},
    },
    "tools": [
        {
            "type": "function",
            "name": "set_lamp",
            "description": "Turn the living-room reading lamp on or off, and set its color as HSV. A color turns it on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {"type": "boolean"},
                    "hue": {"type": "integer", "minimum": 0, "maximum": 360},
                    "saturation": {"type": "integer", "minimum": 0, "maximum": 100},
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["on"],
            },
        },
        {
            "type": "function",
            "name": "wait_for_user",
            "description": "Call this when the audio does not ask to change the reading lamp. Ends the turn with no spoken reply.",
            "parameters": {"type": "object", "properties": {}},
        },
    ],
}

def load_env():
    path = Path(__file__).with_name(".env")
    if not path.is_file():
        sys.exit(f"missing {path}. Copy .env.example and set OPENAI_API_KEY.")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"Set OPENAI_API_KEY in {path}")

def clamp(value, lo, hi, default):
    if value is None:
        return default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))

async def drive(on, hue=None, saturation=None, brightness=None):
    from kasa import DeviceConfig, Module
    from kasa.iot import IotBulb
    bulb = None
    try:
        bulb = IotBulb(HOST, config=DeviceConfig(host=HOST, timeout=5))
        await bulb.update()
        if not on:
            await bulb.turn_off()
            print("\nlights off", flush=True)
            return {"on": False}
        await bulb.turn_on()
        if hue is None:
            print("\nlights on", flush=True)
            return {"on": True}
        hsv = (clamp(hue, 0, 360, 0), clamp(saturation, 0, 100, 100), clamp(brightness, 0, 100, 100))
        await bulb.modules[Module.Light].set_hsv(*hsv)
        print(f"\nlights on, hsv {hsv[0]} {hsv[1]} {hsv[2]}", flush=True)
        return {"on": True, "hue": hsv[0], "saturation": hsv[1], "brightness": hsv[2]}
    except Exception as exc:
        text = str(exc).strip() or type(exc).__name__
        print(f"\n{text}", flush=True)
        return {"error": text}
    finally:
        if bulb is not None:
            with contextlib.suppress(Exception):
                await bulb.disconnect()

class Speaker:
    def __init__(self):
        import numpy as np, sounddevice as sd
        self.np, self.buf, self.lock = np, bytearray(), threading.Lock()
        self.stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.play, blocksize=RATE // 50)
        self.stream.start()

    def play(self, outdata, frames, _time, _status):
        need = frames * 2
        with self.lock:
            chunk = bytes(self.buf[:need])
            del self.buf[:need]
        outdata[:] = self.np.frombuffer(chunk.ljust(need, b"\x00"), dtype="int16").reshape(-1, 1)

    def write(self, pcm):
        with self.lock:
            self.buf += pcm

    def clear(self):
        with self.lock:
            self.buf.clear()

    def close(self):
        self.stream.stop()
        self.stream.close()

async def send_mic(connection):
    import sounddevice as sd
    loop, queue = asyncio.get_running_loop(), asyncio.Queue()

    def heard(indata, _frames, _time, _status):
        loop.call_soon_threadsafe(queue.put_nowait, indata.tobytes())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=heard, blocksize=RATE // 50):
        while True:
            chunk = await queue.get()
            await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))

async def finish_call(connection, speaker, state, name, call_id, arguments):
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if name == "wait_for_user":
        state["quiet"] = True
        speaker.clear()
        with contextlib.suppress(Exception):
            await connection.response.cancel()
            await connection.output_audio_buffer.clear()
        result = {"wait": True}
        print("\nignored", flush=True)
    else:
        state["quiet"] = False
        result = await drive(bool(args.get("on")), args.get("hue"), args.get("saturation"), args.get("brightness"))
    await connection.conversation.item.create(item={
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result),
    })
    if name != "wait_for_user":
        await connection.response.create(response={
            "tool_choice": "none",
            "instructions": "One short sentence confirming the lamp. Do not call a tool.",
        })

async def main():
    from openai import AsyncOpenAI
    load_env()
    speaker, who, seen, state = Speaker(), None, set(), {"quiet": False}
    async with AsyncOpenAI() as client, client.realtime.connect(model="gpt-realtime-2.1", max_retries=0) as connection:
        loop, mic = asyncio.get_running_loop(), None

        def stop():
            asyncio.create_task(connection.close())

        loop.add_signal_handler(signal.SIGINT, stop)
        await connection.session.update(session=SESSION, event_id="start")
        try:
            async for event in connection:
                kind = event.type
                if kind == "session.updated":
                    print(f"session {getattr(event.session, 'id', 'ready')}. Ctrl-C stops.", flush=True)
                    if mic is None:
                        mic = asyncio.create_task(send_mic(connection))
                elif kind == "input_audio_buffer.speech_started":
                    state["quiet"] = False
                elif kind == "response.output_audio.delta" and not state["quiet"]:
                    speaker.write(base64.b64decode(event.delta))
                elif kind == "response.output_audio_transcript.delta" and event.delta and not state["quiet"]:
                    if who != "stanley":
                        print("\nstanley: ", end="", flush=True)
                        who = "stanley"
                    print(event.delta, end="", flush=True)
                elif kind == "response.function_call_arguments.done" and event.call_id not in seen:
                    seen.add(event.call_id)
                    asyncio.create_task(finish_call(connection, speaker, state, event.name, event.call_id, event.arguments))
                elif kind == "error":
                    print(f"\n{event}", flush=True)
                    if getattr(getattr(event, "error", None), "client_event_id", None) == "start":
                        break
        finally:
            if mic is not None:
                mic.cancel()
            speaker.close()

if __name__ == "__main__":
    asyncio.run(main())
