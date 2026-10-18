"""Listen locally for the wake word, then open one realtime session."""

import asyncio, base64, contextlib, json, os, signal, sys, threading, time, zipfile
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

RATE = 24000
IDLE, FRAME = 12.0, RATE // 50
DISMISS = ("shut up", "goodbye", "good bye", "stop talking", "be quiet", "go away")
WAKE_RATE = 16000
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

def load_home():
    path = Path(__file__).with_name("home.json")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not read {path}: {exc}")
    agent = data.get("agent") or {}
    name = str(agent.get("name") or "").strip()
    wake = str(agent.get("wake") or name).strip()
    voice = str(agent.get("voice") or "").strip()
    aliases = [str(item).strip() for item in (agent.get("wake_aliases") or []) if str(item).strip()]
    try:
        buffer_seconds = float(agent.get("buffer_seconds") or 180)
    except (TypeError, ValueError):
        buffer_seconds = 180
    buffer_seconds = max(5.0, buffer_seconds)
    lights = [light for light in (data.get("lights") or []) if light.get("name") and light.get("host")]
    if not name or not wake or not voice:
        sys.exit(f"{path} needs agent.name, agent.wake, and agent.voice")
    if not lights:
        sys.exit(f"No lights in {path}")
    return {
        "name": name, "wake": wake, "voice": voice, "aliases": aliases,
        "buffer_seconds": buffer_seconds, "lights": lights,
    }

def woke(text, home):
    heard = text.casefold().split()
    for phrase in (home["wake"], *home["aliases"]):
        words = phrase.casefold().split()
        if words and any(heard[i:i + len(words)] == words for i in range(len(heard) - len(words) + 1)):
            return True
    return False

def find_light(lights, name):
    wanted = (name or "").casefold().strip()
    for light in lights:
        if light["name"].casefold() == wanted:
            return light
    hits = [light for light in lights if wanted and wanted in light["name"].casefold()]
    return hits[0] if len(hits) == 1 else None

def light_list(lights):
    return "\n".join(f"- {light['name']}: {light.get('description') or light['host']}" for light in lights)

def realtime_session(home):
    lights = home["lights"]
    name = home["name"]
    listing = light_list(lights)
    return {
        "type": "realtime",
        "model": "gpt-realtime-2.1",
        "output_modalities": ["audio"],
        "instructions": (
            f"You are {name}. You can change these lights, and you can answer when someone is talking to you.\n"
            f"{listing}\n"
            "Call set_light only when they clearly ask to turn one of these on or off or to change its color. "
            "Use that light's name. "
            "Choose hue from 0 to 360, saturation from 0 to 100, and brightness from 0 to 100. "
            "The session started because they said your name. A follow-up, such as 'no, purple' or another question, is still for you. "
            "Answer a question to you in one or two sentences. "
            f"If the audio is a television, other people, or anything not said to you and not a follow-up, call not_for_me and say nothing. "
            "Call end_session when they say goodbye, see you later, okay thanks, shut up, stop talking, be quiet, go away, or that they are done. "
            "Say nothing when you call not_for_me or end_session."
        ),
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": RATE},
                "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": True},
            },
            "output": {"format": {"type": "audio/pcm", "rate": RATE}, "voice": home["voice"]},
        },
        "tools": [
            {
                "type": "function",
                "name": "set_light",
                "description": "Turn one listed light on or off, and set its color as HSV. A color turns it on.\n" + listing,
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
            },
            {
                "type": "function",
                "name": "end_session",
                "description": "The person is done or wants you to stop talking. Goodbye, see you later, okay thanks, shut up, stop talking, be quiet, go away.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "not_for_me",
                "description": f"The audio is not addressed to {name} and is not a follow-up. Television, other people, or background. Say nothing.",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    }

def status(text):
    print(f"\n[{text}]", flush=True)

def load_env():
    paths = [Path(__file__).with_name(".env"), Path(__file__).parents[1] / "gpt-live-lights" / ".env"]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"Set OPENAI_API_KEY in {paths[0]}")

def wake_model():
    root = Path(__file__).with_name(".wake-model")
    path = root / "vosk-model-small-en-us-0.15"
    if path.is_dir():
        return path
    root.mkdir(exist_ok=True)
    archive = root / "model.zip"
    status("downloading the local wake model")
    urlretrieve(MODEL_URL, archive)
    with zipfile.ZipFile(archive) as packed:
        packed.extractall(root)
    archive.unlink(missing_ok=True)
    return path

def clamp(value, lo, hi, default):
    if value is None:
        return default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))

def level(pcm):
    import numpy as np
    samples = np.frombuffer(pcm, dtype=np.int16)
    return float(np.abs(samples).mean()) if len(samples) else 0.0

def to_16k(pcm):
    import numpy as np
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if not len(samples):
        return b""
    count = int(len(samples) * WAKE_RATE / RATE)
    grid = np.linspace(0, len(samples) - 1, count)
    return np.interp(grid, np.arange(len(samples)), samples).astype(np.int16).tobytes()

def utterance(frames):
    # Keep every stretch of talk in the buffer, including across pauses.
    # Drop the quiet, so a conversation can be caught up without uploading silence.
    if not frames:
        return b""
    levels = sorted(value for _pcm, value in frames)
    floor = levels[len(levels) // 5]
    cutoff = max(floor * 3, floor + 40)
    speech = [value >= cutoff for _pcm, value in frames]
    pad = int(0.15 * RATE / FRAME)
    if not any(speech):
        return b"".join(pcm for pcm, _value in frames[-int(20 * RATE / FRAME):])
    keep = [False] * len(frames)
    for index, talking in enumerate(speech):
        if not talking:
            continue
        for nearby in range(max(0, index - pad), min(len(frames), index + pad + 1)):
            keep[nearby] = True
    return b"".join(pcm for (pcm, _value), wanted in zip(frames, keep) if wanted)

def bloop(rising):
    import numpy as np
    low, high = (520, 780) if rising else (780, 520)

    def note(freq):
        count = int(RATE * 0.09)
        t = np.linspace(0, 0.09, count, endpoint=False)
        edge = np.minimum(t / 0.02, (0.09 - t) / 0.02)
        wave = 0.35 * np.sin(2 * np.pi * freq * t) * np.clip(edge, 0, 1)
        return (wave * 32767).astype(np.int16).tobytes()

    return note(low) + bytes(int(RATE * 2 * 0.04)) + note(high)

async def drive(lights, name, on, hue=None, saturation=None, brightness=None):
    from kasa import Discover, Module
    light = find_light(lights, name)
    if light is None:
        print(f"\nno light named {name}", flush=True)
        return {"error": f"no light named {name}"}
    device = None
    try:
        device = await Discover.discover_single(light["host"], timeout=5)
        await device.update()
        label = light["name"]
        if not on:
            await device.turn_off()
            print(f"\n{label} off", flush=True)
            return {"name": label, "on": False}
        await device.turn_on()
        if hue is None:
            print(f"\n{label} on", flush=True)
            return {"name": label, "on": True}
        bulb = device.modules.get(Module.Light)
        if bulb is None:
            print(f"\n{label} has no color", flush=True)
            return {"name": label, "on": True, "error": "no color"}
        hsv = (clamp(hue, 0, 360, 0), clamp(saturation, 0, 100, 100), clamp(brightness, 0, 100, 100))
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

class Speaker:
    def __init__(self):
        import numpy as np, sounddevice as sd
        self.np, self.buf, self.lock, self.until = np, bytearray(), threading.Lock(), 0.0
        self.stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.play, blocksize=FRAME)
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
            now = time.monotonic()
            start = self.until if self.until > now else now
            self.until = start + len(pcm) / (RATE * 2)

    def playing(self):
        with self.lock:
            return time.monotonic() < self.until

    def clear(self):
        with self.lock:
            self.buf.clear()
            self.until = 0.0

    def close(self):
        self.stream.stop()
        self.stream.close()

class Ear:
    def __init__(self, loop, speaker, buffer_seconds):
        import sounddevice as sd
        self.loop, self.speaker = loop, speaker
        self.ring, self.lock = deque(maxlen=int(buffer_seconds * RATE / FRAME)), threading.Lock()
        self.mode, self.backlog = "wake", b""
        self.wake, self.live, self.dismiss = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.heard, blocksize=FRAME)
        self.stream.start()

    def heard(self, indata, _frames, _time, _status):
        pcm = indata.tobytes()
        with self.lock:
            self.ring.append((pcm, level(pcm)))
            mode = self.mode
        if mode == "wake":
            self.loop.call_soon_threadsafe(self.wake.put_nowait, pcm)
        else:
            self.loop.call_soon_threadsafe(self.dismiss.put_nowait, pcm)
            self.loop.call_soon_threadsafe(self.live.put_nowait, pcm)

    def engage(self):
        with self.lock:
            frames = list(self.ring)
            self.mode = "session"
        self.backlog = utterance(frames)
        return self.backlog

    def idle(self):
        with self.lock:
            self.mode = "wake"
            self.ring.clear()
        for queue in (self.wake, self.live, self.dismiss):
            while not queue.empty():
                queue.get_nowait()

    def close(self):
        self.stream.stop()
        self.stream.close()

async def wait_for_wake(ear, recognizer, shutdown, home):
    pending = b""
    while not shutdown.is_set():
        try:
            pcm = await asyncio.wait_for(ear.wake.get(), 0.2)
        except asyncio.TimeoutError:
            continue
        pending += to_16k(pcm)
        if len(pending) < WAKE_RATE * 2 // 5:
            continue
        heard = recognizer.AcceptWaveform(pending)
        pending = b""
        payload = recognizer.Result() if heard else recognizer.PartialResult()
        text = json.loads(payload).get("text" if heard else "partial", "")
        if woke(text, home):
            return

async def send_audio(connection, ear):
    audio = ear.backlog
    step = FRAME * 2
    for offset in range(0, len(audio), step):
        chunk = audio[offset:offset + step]
        await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))
    while True:
        chunk = await ear.live.get()
        await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))

async def finish_call(connection, speaker, state, name, call_id, arguments):
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if name == "not_for_me":
        state["quiet"] = True
        speaker.clear()
        result = {"ignored": True}
        print("\nnot for me", flush=True)
    elif name == "end_session":
        result = {"ended": True}
        state["stop"] = "goodbye"
    else:
        state["quiet"] = False
        state["busy"] = True
        state["last_engage"] = time.monotonic()
        result = await drive(state["lights"], args.get("name"), bool(args.get("on")), args.get("hue"), args.get("saturation"), args.get("brightness"))
        state["busy"] = False
        state["last_engage"] = time.monotonic()
    await connection.conversation.item.create(item={
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result),
    })
    if name == "end_session":
        await connection.close()
    elif name != "not_for_me":
        await request_reply(connection, state)

CONFIRM = {
    "tool_choice": "none",
    "instructions": "One short sentence confirming the light. Do not call a tool.",
}

async def request_reply(connection, state):
    async with state["lock"]:
        if state["responding"]:
            state["confirm"] = True
            return
    await connection.response.create(response=CONFIRM)

def dismissal(text):
    return next((phrase for phrase in DISMISS if phrase in text), None)

async def watch_dismiss(ear, speaker, connection, state, model):
    from vosk import KaldiRecognizer
    recognizer, pending = KaldiRecognizer(model, WAKE_RATE), b""
    while not state["stop"]:
        try:
            pcm = await asyncio.wait_for(ear.dismiss.get(), 0.2)
        except asyncio.TimeoutError:
            continue
        pending += to_16k(pcm)
        if len(pending) < WAKE_RATE * 2 // 5:
            continue
        heard = recognizer.AcceptWaveform(pending)
        pending = b""
        payload = recognizer.Result() if heard else recognizer.PartialResult()
        text = json.loads(payload).get("text" if heard else "partial", "")
        phrase = dismissal(text)
        if phrase is None:
            continue
        state["stop"] = phrase
        speaker.clear()
        await connection.close()
        return

async def watch_idle(connection, speaker, state):
    while not state["stop"]:
        await asyncio.sleep(0.5)
        if state["busy"] or speaker.playing():
            state["last_engage"] = time.monotonic()
        elif time.monotonic() - state["last_engage"] > IDLE:
            state["stop"] = "idle"
            await connection.close()
            return

async def session(client, ear, speaker, started, current, model, home):
    who, seen = None, set()
    state = {
        "quiet": False, "busy": False, "stop": None, "responding": False, "confirm": False,
        "drop_audio": False, "last_engage": time.monotonic(), "lock": asyncio.Lock(),
        "lights": home["lights"],
    }
    mic = watch = dismiss = None
    async with client.realtime.connect(model="gpt-realtime-2.1", max_retries=0) as connection:
        current["connection"] = connection
        dismiss = asyncio.create_task(watch_dismiss(ear, speaker, connection, state, model))
        await connection.session.update(session=realtime_session(home), event_id="start")
        try:
            async for event in connection:
                kind = event.type
                if kind == "session.updated" and mic is None:
                    state["last_engage"] = time.monotonic()
                    status(f"engaged, session ready in {time.monotonic() - started:.1f}s, {getattr(event.session, 'id', 'ready')}")
                    mic = asyncio.create_task(send_audio(connection, ear))
                    watch = asyncio.create_task(watch_idle(connection, speaker, state))
                elif kind == "input_audio_buffer.speech_started":
                    state["quiet"] = False
                    state["drop_audio"] = True
                    speaker.clear()
                elif kind == "response.created":
                    state["responding"] = True
                    state["drop_audio"] = False
                elif kind in ("response.done", "response.cancelled"):
                    async with state["lock"]:
                        state["responding"] = False
                        confirm = state["confirm"]
                        state["confirm"] = False
                    if confirm and not state["stop"]:
                        await connection.response.create(response=CONFIRM)
                elif kind == "response.output_audio.delta" and not state["quiet"] and not state["drop_audio"]:
                    state["last_engage"] = time.monotonic()
                    speaker.write(base64.b64decode(event.delta))
                elif kind == "response.output_audio_transcript.delta" and event.delta and not state["quiet"]:
                    if who != home["name"]:
                        print(f"\n{home['name']}: ", end="", flush=True)
                        who = home["name"]
                    print(event.delta, end="", flush=True)
                elif kind == "response.function_call_arguments.done" and event.call_id not in seen:
                    seen.add(event.call_id)
                    asyncio.create_task(finish_call(connection, speaker, state, event.name, event.call_id, event.arguments))
                elif kind == "error":
                    message = getattr(getattr(event, "error", None), "message", "") or ""
                    if "active response in progress" in message:
                        async with state["lock"]:
                            state["responding"] = True
                            state["confirm"] = True
                        continue
                    print(f"\n{event}", flush=True)
                    if getattr(getattr(event, "error", None), "client_event_id", None) == "start":
                        state["stop"] = "error"
                        break
        finally:
            current["connection"] = None
            for task in (mic, watch, dismiss):
                if task is not None:
                    task.cancel()
    return state["stop"] or "closed"

async def main():
    from openai import AsyncOpenAI
    from vosk import KaldiRecognizer, Model
    load_env()
    home = load_home()
    model = Model(str(wake_model()))
    loop, shutdown, current = asyncio.get_running_loop(), asyncio.Event(), {"connection": None}
    speaker, ear = Speaker(), None

    def stop():
        shutdown.set()
        connection = current["connection"]
        if connection is not None:
            asyncio.create_task(connection.close())

    loop.add_signal_handler(signal.SIGINT, stop)
    ear = Ear(loop, speaker, home["buffer_seconds"])
    try:
        async with AsyncOpenAI() as client:
            while not shutdown.is_set():
                status(f"listening for {home['wake']}")
                recognizer = KaldiRecognizer(model, WAKE_RATE)
                await wait_for_wake(ear, recognizer, shutdown, home)
                if shutdown.is_set():
                    break
                started = time.monotonic()
                audio = ear.engage()
                status(f"heard {home['wake']}")
                speaker.write(bloop(True))
                status(f"opening session, {len(audio) / (RATE * 2):.1f}s buffered")
                try:
                    reason = await session(client, ear, speaker, started, current, model, home)
                except Exception as exc:
                    reason = str(exc).strip() or type(exc).__name__
                if shutdown.is_set():
                    break
                status(f"disengaged ({reason})")
                speaker.write(bloop(False))
                await asyncio.sleep(0.5)
                ear.idle()
    finally:
        if ear is not None:
            ear.close()
        speaker.close()

if __name__ == "__main__":
    asyncio.run(main())
