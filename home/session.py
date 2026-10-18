"""Wake word, then one realtime session. A due reminder can open one too."""

import asyncio, base64, json, signal, threading, time, zipfile
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

from .config import ROOT, load_env, load_home
from .tools import find_tool, load_tools
from .tools.timers import announce_text, spoken_line
from .transcript import Log

RATE = 24000
REALTIME_MODEL = "gpt-realtime-2.1-mini"
IDLE, FRAME = 20.0, RATE // 50
WAKE_RATE = 16000
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
CONFIRM = {
    "tool_choice": "none",
    "instructions": "One short sentence confirming what you just did. Do not call a tool.",
}

def status(text):
    print(f"\n[{text}]", flush=True)

def woke(text, home):
    heard = text.casefold().split()
    for phrase in (home["wake"], *home["aliases"]):
        words = phrase.casefold().split()
        if words and any(heard[i:i + len(words)] == words for i in range(len(heard) - len(words) + 1)):
            return True
    return False

def wake_model():
    root = ROOT / ".wake-model"
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

def realtime_session(home, tools, reminder=None, memory=""):
    name = home["name"]
    abilities = [tool.instructions(home) for tool in tools]
    abilities = [text for text in abilities if text]
    parts = [f"You are {name}. You can answer when someone is talking to you."]
    if memory:
        parts.append(memory)
    parts.extend(abilities)
    if reminder:
        line = spoken_line(reminder)
        parts.append(
            f"The thing you are telling them is: {line} "
            "If they ask what you meant or what you are talking about, say that sentence again. "
            "Do not describe a reminder, and do not apologize instead of saying it. "
            "If they sound finished, call end_session."
        )
        parts.append("Quiet or the beep is not not_for_me. After you have said it, television or other people is not_for_me and you say nothing.")
    else:
        parts.append("The session started because they said your name. A follow-up, such as 'no, purple' or another question, is still for you.")
        parts.append("If the audio is a television, other people, or anything not said to you and not a follow-up, call not_for_me and say nothing.")
    parts.extend([
        "Answer a question to you in one or two sentences.",
        "You decide when the conversation is over. When they sound finished, or they want you to stop, call end_session. "
        "Examples of that sound, not a complete list: great thanks, thanks, goodbye, okay great, the deal thanks Stanley, shut up. "
        "Other words with that same meaning count. After a reminder, that reply means they got it. "
        "Call the tool. Going quiet without end_session leaves the session open, which is wrong. Say nothing else when you call it.",
        "Say nothing when you call not_for_me.",
    ])
    instructions = "\n".join(parts)
    schemas = [schema for tool in tools for schema in tool.schemas(home)]
    schemas += [
        {
            "type": "function",
            "name": "end_session",
            "description": "The person sounds finished, or wants you to stop. You judge it. Examples: great thanks, thanks, goodbye, okay great, the deal thanks Stanley. Same meaning in other words counts too.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "not_for_me",
            "description": f"The audio is not addressed to {name} and is not a follow-up. Television, other people, or background. Say nothing.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    return {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "output_modalities": ["audio"],
        "instructions": instructions,
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": RATE},
                "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": True},
                **({"transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"}} if home.get("transcript_minutes") else {}),
            },
            "output": {"format": {"type": "audio/pcm", "rate": RATE}, "voice": home["voice"]},
        },
        "tools": schemas,
    }

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
        self.wake, self.live = asyncio.Queue(), asyncio.Queue()
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
            self.loop.call_soon_threadsafe(self.live.put_nowait, pcm)

    def engage(self, keep=True):
        with self.lock:
            frames = list(self.ring)
            self.mode = "session"
        self.backlog = utterance(frames) if keep else b""
        return self.backlog

    def idle(self):
        with self.lock:
            self.mode = "wake"
            self.ring.clear()
        for queue in (self.wake, self.live):
            while not queue.empty():
                queue.get_nowait()

    def close(self):
        self.stream.stop()
        self.stream.close()

async def wait_for_wake(ear, recognizer, shutdown, home, book):
    pending = b""
    while not shutdown.is_set():
        if book.due():
            return "timer"
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
            return "wake"
    return None

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
    opening = state.get("opening")
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if name == "set_timer" and opening:
        result = {"error": "That reminder is already running. Tell them it is time. Do not call set_timer."}
        print("\nalready running", flush=True)
    elif name == "not_for_me" and state.get("needs_announce") and not state.get("announced") and not state.get("got_audio"):
        result = {"error": "A reminder is due. Say it out loud now. Do not call not_for_me."}
        state["quiet"] = False
        async with state["lock"]:
            state["redo_announce"] = True
        print("\nreminder still due", flush=True)
    elif name == "not_for_me":
        state["quiet"] = True
        speaker.clear()
        result = {"ignored": True}
        print("\nnot for me", flush=True)
    elif name == "end_session":
        result = {"ended": True}
        state["stop"] = "goodbye"
    elif name in ("retry_in", "acknowledge"):
        tool = find_tool(state["tools"], name)
        result = await tool.call(name, args, state["home"]) if tool else {"error": f"unknown tool {name}"}
    else:
        tool = find_tool(state["tools"], name)
        state["quiet"] = False
        state["busy"] = True
        state["last_engage"] = time.monotonic()
        if tool is None:
            result = {"error": f"unknown tool {name}"}
            print(f"\nunknown tool {name}", flush=True)
        else:
            result = await tool.call(name, args, state["home"])
        state["busy"] = False
        state["last_engage"] = time.monotonic()
    await connection.conversation.item.create(item={
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result),
    })
    if name == "end_session":
        await connection.close()
    elif state.get("redo_announce") and not state.get("announced") and not state.get("got_audio"):
        async with state["lock"]:
            speak_now = state.get("redo_announce") and not state["responding"]
            if speak_now:
                state["redo_announce"] = False
        if speak_now and not state["stop"] and state.get("announce"):
            await speak(connection, state, state["announce"], state.get("aloud") or state["announce"])
    elif not (name == "set_timer" and opening) and name not in ("not_for_me", "retry_in", "acknowledge"):
        await request_reply(connection, state)

async def request_reply(connection, state):
    async with state["lock"]:
        if state["responding"]:
            state["confirm"] = True
            return
    await connection.response.create(response=CONFIRM)

async def watch_idle(connection, speaker, state):
    while not state["stop"]:
        await asyncio.sleep(0.5)
        if state["busy"] or state.get("responding") or speaker.playing():
            state["last_engage"] = time.monotonic()
        elif time.monotonic() - state["last_engage"] > IDLE:
            state["stop"] = "idle"
            await connection.close()
            return

async def speak(connection, state, instructions, aloud):
    state["needs_announce"] = True
    state["opening"] = True
    state["announce"] = instructions
    state["aloud"] = aloud
    state["announce_tries"] = state.get("announce_tries", 0) + 1
    if state["announce_tries"] > 3:
        return
    async with state["lock"]:
        if state["responding"]:
            state["later"] = instructions
            return
    if not state.get("planted"):
        state["planted"] = True
        await connection.conversation.item.create(item={
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": aloud}],
        })
    state["reminder_text"] = instructions
    await connection.response.create(response={"instructions": instructions})

async def listen_after(ear, speaker, connection, state):
    while not state["stop"] and speaker.playing():
        await asyncio.sleep(0.05)
    while True:
        try:
            ear.live.get_nowait()
        except asyncio.QueueEmpty:
            break
    if not state["stop"]:
        await send_audio(connection, ear)

async def watch_due(connection, state, book):
    while not state["stop"]:
        await asyncio.sleep(0.5)
        due = book.claim_due()
        if not due:
            continue
        names = ", ".join(timer["what"] for timer in due)
        status(f"reminder due: {names}")
        await speak(connection, state, announce_text(due), spoken_line(due))

def heard_back(reason):
    return reason == "goodbye"

def remember(log, who, text):
    log.add(who, text)
    if who == "person" and text.strip():
        print(f"\nPerson: {text.strip()}", flush=True)

async def run_session(client, ear, speaker, started, current, home, tools, book, log, reminder=None):
    who, seen = None, set()
    state = {
        "quiet": False, "busy": False, "stop": None, "responding": False, "confirm": False,
        "drop_audio": False, "last_engage": time.monotonic(), "lock": asyncio.Lock(),
        "home": home, "tools": tools, "later": None, "reminder_text": None,
        "needs_announce": False, "announced": False, "announce": None,
        "announce_tries": 0, "got_audio": False, "redo_announce": False, "opening": False,
        "line": "", "logged_line": False, "planted": False,
    }
    mic = watch = due = None
    opened = False
    async with client.realtime.connect(model=REALTIME_MODEL, max_retries=0) as connection:
        current["connection"] = connection
        await connection.session.update(
            session=realtime_session(home, tools, reminder, log.prompt(home["name"])), event_id="start",
        )
        try:
            async for event in connection:
                kind = event.type
                if kind == "session.updated" and not opened:
                    opened = True
                    state["last_engage"] = time.monotonic()
                    status(f"engaged, session ready in {time.monotonic() - started:.1f}s, {getattr(event.session, 'id', 'ready')}")
                    if reminder:
                        await speak(connection, state, announce_text(reminder), spoken_line(reminder))
                    else:
                        mic = asyncio.create_task(send_audio(connection, ear))
                    watch = asyncio.create_task(watch_idle(connection, speaker, state))
                    due = asyncio.create_task(watch_due(connection, state, book))
                elif kind == "input_audio_buffer.speech_started":
                    if state.get("needs_announce") and not state.get("announced"):
                        continue
                    state["quiet"] = False
                    state["drop_audio"] = True
                    state["last_engage"] = time.monotonic()
                    speaker.clear()
                elif kind == "response.created":
                    if state["line"] and not state["logged_line"]:
                        remember(log, "assistant", state["line"])
                    state["line"], state["logged_line"] = "", False
                    state["responding"] = True
                    state["last_engage"] = time.monotonic()
                    state["drop_audio"] = False
                    state["got_audio"] = False
                    state["reminder_text"] = None
                elif kind in ("response.done", "response.cancelled"):
                    if state["line"] and not state["logged_line"]:
                        remember(log, "assistant", state["line"])
                        state["logged_line"] = True
                    state["line"] = ""
                    await asyncio.sleep(0)
                    async with state["lock"]:
                        state["responding"] = False
                        confirm = state["confirm"]
                        state["confirm"] = False
                        later = state["later"]
                        state["later"] = None
                        spoke = state.get("got_audio")
                        if spoke and state.get("needs_announce"):
                            state["announced"] = True
                            state["opening"] = False
                        redo = state.get("redo_announce") and not spoke and not state.get("announced")
                        state["redo_announce"] = False
                    if state["stop"]:
                        continue
                    follow = later or (state.get("announce") if redo else None)
                    if follow:
                        await speak(connection, state, follow, state.get("aloud") or follow)
                    elif confirm:
                        await connection.response.create(response=CONFIRM)
                    if state.get("announced") and mic is None and not follow:
                        mic = asyncio.create_task(listen_after(ear, speaker, connection, state))
                elif kind == "response.output_audio.delta" and not state["quiet"] and not state["drop_audio"]:
                    if state.get("needs_announce"):
                        state["got_audio"] = True
                    state["last_engage"] = time.monotonic()
                    speaker.write(base64.b64decode(event.delta))
                elif kind == "response.output_audio_transcript.delta" and event.delta and not state["quiet"]:
                    state["line"] += event.delta
                    if who != home["name"]:
                        print(f"\n{home['name']}: ", end="", flush=True)
                        who = home["name"]
                    print(event.delta, end="", flush=True)
                elif kind == "response.output_audio_transcript.done":
                    text = getattr(event, "transcript", "") or state["line"]
                    if not state["logged_line"]:
                        remember(log, "assistant", text)
                        state["logged_line"] = True
                    state["line"] = ""
                elif kind == "conversation.item.input_audio_transcription.completed":
                    remember(log, "person", getattr(event, "transcript", "") or "")
                elif kind == "response.function_call_arguments.done" and event.call_id not in seen:
                    seen.add(event.call_id)
                    asyncio.create_task(finish_call(connection, speaker, state, event.name, event.call_id, event.arguments))
                elif kind == "error":
                    message = getattr(getattr(event, "error", None), "message", "") or ""
                    if "active response in progress" in message:
                        async with state["lock"]:
                            state["responding"] = True
                            if state["reminder_text"]:
                                state["later"] = state["reminder_text"]
                            else:
                                state["confirm"] = True
                        continue
                    print(f"\n{event}", flush=True)
                    if getattr(getattr(event, "error", None), "client_event_id", None) == "start":
                        state["stop"] = "error"
                        break
        finally:
            current["connection"] = None
            for task in (mic, watch, due):
                if task is not None:
                    task.cancel()
    return state["stop"] or "closed", bool(state.get("announced") or not reminder)

async def main():
    from openai import AsyncOpenAI
    from vosk import KaldiRecognizer, Model
    load_env()
    home, tools = load_home(), load_tools()
    book, log = find_tool(tools, "set_timer").book, Log(home["transcript_minutes"])
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
                status(f"listening for {home['wake']}{book.next_phrase()}")
                recognizer = KaldiRecognizer(model, WAKE_RATE)
                kind = await wait_for_wake(ear, recognizer, shutdown, home, book)
                if shutdown.is_set() or kind is None:
                    break
                started = time.monotonic()
                reminder = book.claim_due() if kind == "timer" else None
                if kind == "timer" and not reminder:
                    continue
                audio = ear.engage(keep=kind != "timer")
                if reminder:
                    names = ", ".join(timer["what"] for timer in reminder)
                    status(f"reminder due: {names}")
                else:
                    status(f"heard {home['wake']}")
                speaker.write(bloop(True))
                if reminder:
                    status("opening session for a reminder")
                else:
                    status(f"opening session, {len(audio) / (RATE * 2):.1f}s buffered")
                said = True
                try:
                    reason, said = await run_session(
                        client, ear, speaker, started, current, home, tools, book, log, reminder,
                    )
                except Exception as exc:
                    reason = str(exc).strip() or type(exc).__name__
                    said = False
                if not said:
                    book.unspoken(list(book.session_ids))
                if heard_back(reason):
                    book.finish(list(book.session_ids))
                else:
                    book.fallback(list(book.session_ids))
                book.session_ids.clear()
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
