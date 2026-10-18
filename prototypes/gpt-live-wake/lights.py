"""Listen locally for Stanley, then open one realtime session."""

import asyncio, base64, contextlib, json, os, signal, sys, threading, time, zipfile
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

HOST, RATE = "192.168.4.38", 24000
BUFFER, IDLE, FRAME = 12.0, 12.0, RATE // 50
DISMISS = ("shut up", "goodbye", "good bye", "stop talking", "be quiet", "go away")
WAKE_RATE = 16000
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
SESSION = {
    "type": "realtime",
    "model": "gpt-realtime-2.1",
    "output_modalities": ["audio"],
    "instructions": (
        "You are Stanley. You can change the living-room reading lamp, and you can answer when someone is talking to you. "
        "Call set_lamp only when they clearly ask to turn that lamp on or off or to change its color. "
        "Choose hue from 0 to 360, saturation from 0 to 100, and brightness from 0 to 100. "
        "The session started because they said your name. A follow-up, such as 'no, purple' or another question, is still for you. "
        "Answer a question to you in one or two sentences. "
        "If the audio is a television, other people, or anything not said to you and not a follow-up, call not_for_me and say nothing. "
        "Call end_session when they say goodbye, see you later, okay thanks, shut up, stop talking, be quiet, go away, or that they are done. "
        "Say nothing when you call not_for_me or end_session."
    ),
    "tool_choice": "auto",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": RATE},
            "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": False},
        },
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
            "name": "end_session",
            "description": "The person is done or wants you to stop talking. Goodbye, see you later, okay thanks, shut up, stop talking, be quiet, go away.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "not_for_me",
            "description": "The audio is not addressed to Stanley and is not a follow-up. Television, other people, or background. Say nothing.",
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
    # The pause after "Stanley" is what lets the spotter finish the word.
    # Keep the speech before that pause, and drop only a long quiet stretch before the sentence.
    if not frames:
        return b""
    levels = sorted(value for _pcm, value in frames)
    floor = levels[len(levels) // 5]
    cutoff = max(floor * 3, floor + 40)
    speech = [value >= cutoff for _pcm, value in frames]
    hold, pad = int(1.5 * RATE / FRAME), int(0.2 * RATE / FRAME)
    if not any(speech):
        return b"".join(pcm for pcm, _value in frames[-int(8 * RATE / FRAME):])
    end = len(frames)
    while end > 0 and not speech[end - 1]:
        end -= 1
    start, gap = end, 0
    for index in range(end - 1, -1, -1):
        if speech[index]:
            start, gap = index, 0
        else:
            gap += 1
            if gap >= hold:
                break
    start, end = max(0, start - pad), min(len(frames), end + pad)
    return b"".join(pcm for pcm, _value in frames[start:end])

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
        self.np, self.buf, self.lock, self.until = np, bytearray(), threading.Lock(), 0.0
        self.stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.play, blocksize=FRAME)
        self.stream.start()

    def play(self, outdata, frames, _time, _status):
        need = frames * 2
        with self.lock:
            chunk = bytes(self.buf[:need])
            del self.buf[:need]
        outdata[:] = self.np.frombuffer(chunk.ljust(need, b"\x00"), dtype="int16").reshape(-1, 1)

    def write(self, pcm, mute=0.35):
        with self.lock:
            self.buf += pcm
            if mute is None:
                return
            now = time.monotonic()
            start = self.until if self.until > now else now
            self.until = start + len(pcm) / (RATE * 2) + mute

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
    def __init__(self, loop, speaker):
        import sounddevice as sd
        self.loop, self.speaker = loop, speaker
        self.ring, self.lock = deque(maxlen=int(BUFFER * RATE / FRAME)), threading.Lock()
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
            if not self.speaker.playing():
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

async def wait_for_stanley(ear, recognizer, shutdown):
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
        if "stanley" in text.split() or "stan lee" in text:
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
        result = await drive(bool(args.get("on")), args.get("hue"), args.get("saturation"), args.get("brightness"))
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
        await connection.response.create(response={
            "tool_choice": "none",
            "instructions": "One short sentence confirming the lamp. Do not call a tool.",
        })

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

async def session(client, ear, speaker, started, current, model):
    who, seen, state = None, set(), {"quiet": False, "busy": False, "stop": None, "last_engage": time.monotonic()}
    mic = watch = dismiss = None
    async with client.realtime.connect(model="gpt-realtime-2.1", max_retries=0) as connection:
        current["connection"] = connection
        dismiss = asyncio.create_task(watch_dismiss(ear, speaker, connection, state, model))
        await connection.session.update(session=SESSION, event_id="start")
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
                elif kind == "response.output_audio.delta" and not state["quiet"]:
                    state["last_engage"] = time.monotonic()
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
    model = Model(str(wake_model()))
    loop, shutdown, current = asyncio.get_running_loop(), asyncio.Event(), {"connection": None}
    speaker, ear = Speaker(), None

    def stop():
        shutdown.set()
        connection = current["connection"]
        if connection is not None:
            asyncio.create_task(connection.close())

    loop.add_signal_handler(signal.SIGINT, stop)
    ear = Ear(loop, speaker)
    try:
        async with AsyncOpenAI() as client:
            while not shutdown.is_set():
                status("listening for Stanley")
                recognizer = KaldiRecognizer(model, WAKE_RATE)
                await wait_for_stanley(ear, recognizer, shutdown)
                if shutdown.is_set():
                    break
                started = time.monotonic()
                audio = ear.engage()
                status("heard Stanley")
                speaker.write(bloop(True), mute=None)
                status(f"opening session, {len(audio) / (RATE * 2):.1f}s buffered")
                try:
                    reason = await session(client, ear, speaker, started, current, model)
                except Exception as exc:
                    reason = str(exc).strip() or type(exc).__name__
                if shutdown.is_set():
                    break
                status(f"disengaged ({reason})")
                speaker.write(bloop(False), mute=None)
                await asyncio.sleep(0.5)
                ear.idle()
    finally:
        if ear is not None:
            ear.close()
        speaker.close()

if __name__ == "__main__":
    asyncio.run(main())
