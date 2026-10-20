"""Wake word, then one realtime session. A due reminder can open one too."""

import asyncio, base64, json, os, signal, threading, time, zipfile
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

from .config import DATA, ROOT, load_env, load_home
from .log import begin, record
from .memory import Bank, Trace, action_line, recall_note
from .preferences import prompt_block, read_notes, revise
from .voice import Voices
from .tools import find_tool, load_tools
from .tools.timers import announce_text, spoken_line
from .transcript import Log

RATE = 24000
REALTIME_MODEL = "grok-voice-3.0"
XAI_BASE = "https://api.x.ai/v1"
IDLE, FRAME = 20.0, RATE // 50
WAKE_RATE = 16000
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
CONFIRM = {
    "tool_choice": "none",
    "instructions": "One short sentence confirming what you just did. Do not call a tool.",
}

def status(text):
    print(f"\n[{text}]", flush=True)
    kind = "status"
    if text.startswith("engaged"):
        kind = "engage"
    elif text.startswith("disengaged"):
        kind = "disengage"
    record(kind, text)

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

def utterance(frames, pause_seconds):
    """Audio since the last quiet gap of pause_seconds. Earlier talk stays out."""
    if not frames:
        return b""
    levels = [value for _pcm, value in frames]
    ordered = sorted(levels)
    # A qualifying pause is long enough to hold this sample, so the floor is the
    # quiet itself and not the speech around it.
    half_pause = max(1, int(0.5 * float(pause_seconds) * RATE / FRAME))
    floor = ordered[min(len(ordered) // 20, half_pause - 1)]
    cutoff = max(floor * 3, floor + 40)
    speech = [value >= cutoff for value in levels]
    blip = int(0.35 * RATE / FRAME)
    index = 0
    while index < len(speech):
        if not speech[index]:
            index += 1
            continue
        end = index + 1
        while end < len(speech) and speech[end]:
            end += 1
        if end - index < blip:
            speech[index:end] = [False] * (end - index)
        index = end
    if not any(speech):
        return b"".join(pcm for pcm, _value in frames[-int(20 * RATE / FRAME):])
    pause_frames = max(1, int(float(pause_seconds) * RATE / FRAME))
    pad = int(0.15 * RATE / FRAME)
    cut, quiet, index = 0, 0, len(speech) - 1
    while index >= 0:
        if speech[index]:
            quiet = 0
            index -= 1
            continue
        run_end = index + 1
        while index >= 0 and not speech[index]:
            index -= 1
            quiet += 1
        if quiet >= pause_frames:
            cut = run_end
            break
        quiet = 0
    start = max(0, cut - pad)
    latest = max(0, len(frames) - int(2 * RATE / FRAME))
    start = min(start, latest)
    return b"".join(pcm for pcm, _value in frames[start:])

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

def speaker_line(name):
    if name and name != "Someone":
        return (
            f"You are talking to {name}. Their voice matched. "
            f"If they ask who they are, or say you do not know them, say {name}. "
            f"Do not say you cannot place their voice. "
            f"A shower with no temperature and no preset uses {name}'s preference."
        )
    return (
        "You do not recognize this voice. Do not guess a name. "
        "If they ask who they are, say you do not recognize them."
    )


def realtime_session(home, tools, reminder=None, memory="", hold_reply=False):
    name = home["name"]
    abilities = [tool.instructions(home) for tool in tools]
    abilities = [text for text in abilities if text]
    parts = [f"You are {name}. You can answer when someone is talking to you."]
    if memory:
        parts.append(memory)
    parts.extend(abilities)
    parts.append(
        "You have a memory. When this session ends, Home writes notes from it: what was said, what you did, and code that mattered. "
        "A later session can place a matching note in front of you. That note is something you remember. The From line is when it was said. "
        "If they ask you to remember something, say that you will. It is written down when the session ends. "
        "A note that shows up answers what was just asked. Say it out loud, even if you just said you did not know. "
        "Say you just remembered, then the fact. Do not leave a wrong answer standing, and do not wait to be asked again. "
        "Do not say you have no memory. Do not invent a note you were not given. There is no tool to call for this."
    )
    people = prompt_block(home.get("people") or [], read_notes(DATA / "people", home.get("people") or []))
    if people:
        parts.append(people)
        parts.append(
            "Home names the speaker from their voice before you answer. "
            "The instructions on your reply say who is speaking. That is who you are talking to. "
            "Use their name and their preferences. Do not say you cannot tell voices apart."
        )
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
    audio_in = {"format": {"type": "audio/pcm", "rate": RATE}}
    if home.get("transcript_minutes"):
        terms = [name, *(home.get("aliases") or [])]
        terms += [item.get("name") for item in (home.get("lights") or [])]
        terms += [item.get("name") for item in (home.get("showers") or [])]
        terms = [str(term)[:50] for term in terms if term][:100]
        audio_in["transcription"] = {"model": "grok-transcribe", "language_hint": "en", "keyterms": terms}
    return {
        "voice": home["voice"],
        "instructions": instructions,
        "turn_detection": {
            "type": "server_vad",
            "create_response": not hold_reply,
            "interrupt_response": True,
        },
        "reasoning": {"effort": "none"},
        "audio": {
            "input": audio_in,
            "output": {"format": {"type": "audio/pcm", "rate": RATE}},
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
        self.heard_audio, self.speech_mark, self.session_bytes = bytearray(), 0, 0
        self.wake, self.live = asyncio.Queue(), asyncio.Queue()
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.heard, blocksize=FRAME)
        self.stream.start()

    def heard(self, indata, _frames, _time, _status):
        pcm = indata.tobytes()
        with self.lock:
            self.ring.append((pcm, level(pcm)))
            mode = self.mode
            if mode == "session":
                self._keep(pcm)
        if mode == "wake":
            self.loop.call_soon_threadsafe(self.wake.put_nowait, pcm)
        else:
            self.loop.call_soon_threadsafe(self.live.put_nowait, pcm)

    def engage(self, keep=True, pause_seconds=8):
        with self.lock:
            frames = list(self.ring)
            self.mode = "session"
            self.heard_audio, self.speech_mark, self.session_bytes = bytearray(), 0, 0
        self.backlog = utterance(frames, pause_seconds) if keep else b""
        if self.backlog:
            with self.lock:
                self._keep(self.backlog)
        return self.backlog

    def idle(self):
        with self.lock:
            self.mode = "wake"
            self.ring.clear()
            self.heard_audio, self.speech_mark, self.session_bytes = bytearray(), 0, 0
        for queue in (self.wake, self.live):
            while not queue.empty():
                queue.get_nowait()

    def close(self):
        self.stream.stop()
        self.stream.close()

    def _keep(self, pcm):
        self.session_bytes += len(pcm)
        self.heard_audio.extend(pcm)
        extra = len(self.heard_audio) - 60 * RATE * 2
        if extra > 0:
            del self.heard_audio[:extra]
            self.speech_mark = max(0, self.speech_mark - extra)

    def mark_at(self, audio_ms):
        with self.lock:
            absolute = int(float(audio_ms) * RATE * 2 / 1000)
            origin = self.session_bytes - len(self.heard_audio)
            self.speech_mark = max(0, min(len(self.heard_audio), absolute - origin))

    def spoken(self):
        with self.lock:
            audio = bytes(self.heard_audio)
            mark = self.speech_mark
        if mark <= 0:
            mark = max(0, len(audio) - 8 * RATE * 2)
        start = max(0, mark - int(0.25 * RATE) * 2)
        clip = audio[start:]
        cap = 12 * RATE * 2
        if len(clip) > cap:
            clip = clip[-cap:]
        return clip

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

# Raw pcm per append. Base64 grows by 4/3, and an append is capped at 15 MiB.
PCM_APPEND = 8 * 1024 * 1024

async def send_pcm(connection, pcm):
    if not pcm:
        return
    for offset in range(0, len(pcm), PCM_APPEND):
        chunk = pcm[offset:offset + PCM_APPEND]
        await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))

async def send_audio(connection, ear):
    # The backlog is already recorded. One append, not a 20ms slice at a time,
    # or 30 seconds of talk takes 30 seconds to upload.
    await send_pcm(connection, ear.backlog)
    while True:
        chunk = await ear.live.get()
        pending = [chunk]
        while True:
            try:
                pending.append(ear.live.get_nowait())
            except asyncio.QueueEmpty:
                break
        await send_pcm(connection, b"".join(pending))

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
    elif name == "retry_in" and state.get("needs_announce") and not state.get("announced") and not state.get("got_audio"):
        result = {"error": "Say the reminder out loud first. Do not call retry_in."}
        state["quiet"] = False
        async with state["lock"]:
            state["redo_announce"] = True
        print("\nsay it first", flush=True)
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
    record("tool", f"{name} {arguments or '{}'} -> {json.dumps(result)}")
    trace = state.get("trace")
    if trace is not None and name not in ("end_session", "not_for_me"):
        trace.did(action_line(name, result))
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

LIVE_NOTE = (
    "Console output just arrived from a script that is still running. "
    "This is information for you to remember, not a turn to talk. Say nothing. "
    "Do not narrate it, summarize it, or acknowledge it. Ordinary progress is not worth speech. "
    "Speak only if this output is an error or something extraordinary they need to hear right now, and then one short sentence. "
    "If they ask what is going on, answer from what the script has printed. Do not call a tool."
)
DONE_NOTE = (
    "A script you started has finished or was stopped. Tell them the output or the error. "
    "If the error is a mistake in your code, call run_script again. "
    "stop_script stops one when they want it stopped. Do not ask them to run anything."
)

def script_response(notes):
    if notes and all("is still running" in note for note in notes):
        return {"instructions": LIVE_NOTE, "tool_choice": "none"}
    return {"instructions": DONE_NOTE}

async def deliver_script(connection, state, text):
    if state.get("stop"):
        return
    async with state["lock"]:
        state.setdefault("notes", []).append(text)
        busy = state["responding"]
    if not busy:
        await flush_notes(connection, state)

async def send_memos(connection, memos):
    await connection.conversation.item.create(item={
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "\n\n".join(memos)}],
    })
    await connection.response.create(response={"instructions": recall_note(memos), "tool_choice": "none"})

async def deliver_memory(connection, state, text):
    if state.get("stop") or not text:
        return
    async with state["lock"]:
        state.setdefault("memos", []).append(text)
        busy = state["responding"]
        if busy:
            return
        memos = state["memos"]
        state["memos"] = []
    await send_memos(connection, memos)

async def surface(bank, connection, state, text):
    try:
        hits = await bank.recall(text)
        if not hits or state.get("stop"):
            return
        seen = state.setdefault("recalled", set())
        fresh = []
        for hit in hits:
            if hit["id"] in seen:
                continue
            seen.add(hit["id"])
            fresh.append(hit["text"])
        if fresh:
            await deliver_memory(connection, state, "\n\n".join(fresh))
    except Exception as exc:
        record("error", f"memory: {exc}")

async def flush_notes(connection, state):
    async with state["lock"]:
        if state.get("responding") or not state.get("notes"):
            return
        notes = state["notes"]
        state["notes"] = []
    await connection.conversation.item.create(item={
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "\n\n".join(notes)}],
    })
    await connection.response.create(response=script_response(notes))

async def request_reply(connection, state):
    async with state["lock"]:
        if state["responding"]:
            state["confirm"] = True
            return
    await connection.response.create(response=CONFIRM)

async def watch_idle(connection, speaker, state):
    while not state["stop"]:
        await asyncio.sleep(0.5)
        if state["busy"] or state.get("responding") or speaker.playing() or state.get("scripts"):
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
    await connection.response.create(response={"instructions": instructions, "tool_choice": "none"})

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
        status(f"reminder due: {spoken_line(due)}")
        await speak(connection, state, announce_text(due), spoken_line(due))

def heard_back(reason):
    return reason == "goodbye"

def remember(log, who, text, name=None):
    log.add(who, text, name)
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    record("person" if who == "person" else "assistant", cleaned)
    return cleaned

def follow(state, who, text):
    if not text:
        return
    trace = state.get("trace")
    if trace is not None:
        speaker = state["home"]["name"] if who == "assistant" else (state.get("speaker") or "Someone")
        trace.say(speaker, text)
    cue = state.get("cue")
    if cue is not None:
        cue(text)

def growth(shown, text):
    cleaned = " ".join((text or "").split())
    if not cleaned or cleaned == shown:
        return cleaned, ""
    if shown and cleaned.startswith(shown):
        return cleaned, cleaned[len(shown):]
    if shown and cleaned.casefold().startswith(shown.casefold()):
        return cleaned, cleaned[len(shown):]
    return cleaned, None

def flush_person(state):
    text = state.get("heard") or ""
    if text and state.get("named_turn") == state.get("turn"):
        show_person(state, text)


def show_person(state, text):
    if state.get("hold_reply") and state.get("named_turn") != state.get("turn"):
        return False
    cleaned, extra = growth(state.get("shown") or "", text)
    if extra is None:
        state["person_open"] = False
        extra = cleaned
    if not extra:
        return False
    label = state.get("speaker") or "Someone"
    if state.get("person_open") and state.get("shown_name") != label:
        state["person_open"] = False
    if not state.get("person_open"):
        print(f"\n{label}: ", end="", flush=True)
        state["person_open"] = True
        state["shown_name"] = label
    print(extra, end="", flush=True)
    state["shown"] = cleaned
    return True

def note_heard(log, state, text):
    cleaned = " ".join((text or "").split())
    if not cleaned or cleaned == state.get("heard_logged"):
        return
    state["heard_logged"] = cleaned
    state["heard"] = ""
    follow(state, "person", remember(log, "person", cleaned, state.get("speaker") or "Someone"))

async def answer_speaker(connection, state, name):
    if state.get("answered_turn") == state.get("turn"):
        return
    state["answered_turn"] = state.get("turn")
    text = speaker_line(name)
    state["speaker_line"] = text
    state["burst"] = False
    if state.get("responding"):
        if state.get("got_audio"):
            return
        state["redo_speaker"] = True
        try:
            await connection.response.cancel()
        except Exception as exc:
            record("error", f"voice: {exc}")
        return
    state["answering"] = True
    try:
        await connection.response.create(response={"instructions": text})
    except Exception as exc:
        record("error", f"voice: {exc}")
        state["redo_speaker"] = True
    finally:
        state["answering"] = False

async def apply_name(voices, ear, state, connection, turn):
    if not voices.gallery:
        state["speaker"] = "Someone"
        return
    await asyncio.sleep(0.4)
    if state.get("turn") != turn or state.get("stop"):
        return
    try:
        name, score, held = await voices.identify(ear.spoken(), current=state.get("speaker"))
    except Exception as exc:
        record("error", f"voice: {exc}")
        name = state.get("speaker") or "Someone"
        score, held = None, name != "Someone"
    if state.get("turn") != turn or state.get("stop"):
        return
    if held and score is not None:
        status(f"voice: {name} {score:.2f} kept")
    elif held:
        status(f"voice: {name} kept")
    else:
        status(f"voice: {name}" if score is None else f"voice: {name} {score:.2f}")
    state["speaker"] = name
    state["named_turn"] = state.get("turn")
    flush_person(state)
    if connection is None or not state.get("hold_reply"):
        return
    if state.get("needs_announce") and not state.get("announced"):
        return
    await answer_speaker(connection, state, name)

async def settle_speaker(state, voices=None):
    task = state.get("naming")
    if task is None:
        return
    first = voices is not None and voices.gallery and voices._model is None
    try:
        await asyncio.wait_for(asyncio.shield(task), 20.0 if first else 2.0)
    except asyncio.TimeoutError:
        pass

def begin_name(voices, ear, state, connection):
    task = state.get("naming")
    if task is not None and not task.done():
        return
    turn = state.get("turn") or 0
    state["naming"] = asyncio.create_task(apply_name(voices, ear, state, connection, turn))

async def run_session(client, ear, speaker, started, current, home, tools, book, log, reminder=None, bank=None, trace=None, voices=None):
    who, seen = None, set()
    state = {
        "quiet": False, "busy": False, "stop": None, "responding": False, "confirm": False,
        "drop_audio": False, "last_engage": time.monotonic(), "lock": asyncio.Lock(),
        "home": home, "tools": tools, "later": None, "reminder_text": None,
        "needs_announce": False, "announced": False, "announce": None,
        "announce_tries": 0, "got_audio": False, "redo_announce": False, "opening": False,
        "line": "", "logged_line": False, "planted": False, "speaker": "Someone", "turn": 0,
        "hold_reply": bool(voices and voices.gallery),
        "reminder": reminder, "memory": log.prompt(home["name"]),
        "notes": [], "scripts": 0,
    }
    mic = watch = due = None
    opened = False
    async with client.realtime.connect(model=REALTIME_MODEL, max_retries=0) as connection:
        current["connection"] = connection
        script_tool = find_tool(tools, "run_script")
        if script_tool is not None:
            script_tool.deliver = lambda text: deliver_script(connection, state, text)
            script_tool.track = state
            if trace is not None:
                script_tool.keep = trace.code
        if trace is not None:
            state["trace"] = trace
        if bank is not None:
            state["cue"] = lambda text: asyncio.create_task(surface(bank, connection, state, text))
        await connection.session.update(
            session=realtime_session(home, tools, reminder, state["memory"], state["hold_reply"]), event_id="start",
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
                    state["turn"] = state.get("turn", 0) + 1
                    state["naming"] = None
                    state["person_open"] = False
                    state["shown"] = ""
                    if not state.get("burst"):
                        started_at = getattr(event, "audio_start_ms", None)
                        if started_at is not None:
                            ear.mark_at(started_at)
                        state["burst"] = True
                elif kind == "input_audio_buffer.speech_stopped" and voices is not None:
                    begin_name(voices, ear, state, connection)
                elif kind == "response.created":
                    state["responding"] = True
                    if state.get("naming") is None and voices is not None and not state.get("hold_reply"):
                        begin_name(voices, ear, state, connection)
                    await settle_speaker(state, voices)
                    if state.get("heard"):
                        if show_person(state, state["heard"]):
                            who = "person"
                        note_heard(log, state, state["heard"])
                    if state["line"] and not state["logged_line"]:
                        follow(state, "assistant", remember(log, "assistant", state["line"]))
                    state["line"], state["logged_line"] = "", False
                    state["responding"] = True
                    state["last_engage"] = time.monotonic()
                    state["drop_audio"] = False
                    state["got_audio"] = False
                    state["reminder_text"] = None
                elif kind in ("response.done", "response.cancelled"):
                    if state["line"] and not state["logged_line"]:
                        follow(state, "assistant", remember(log, "assistant", state["line"]))
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
                        pending = [] if later or redo else list(state.get("notes") or [])
                        if pending:
                            state["notes"] = []
                        memos = [] if later or redo else list(state.get("memos") or [])
                        if memos:
                            state["memos"] = []
                    if state["stop"]:
                        continue
                    if state.get("redo_speaker") and state.get("speaker_line") and not spoke:
                        state["redo_speaker"] = False
                        await connection.response.create(response={"instructions": state["speaker_line"]})
                        continue
                    state["redo_speaker"] = False
                    carry = later or (state.get("announce") if redo else None)
                    if carry:
                        await speak(connection, state, carry, state.get("aloud") or state["announce"])
                    elif pending:
                        if memos:
                            async with state["lock"]:
                                state["memos"] = memos + list(state.get("memos") or [])
                        await connection.conversation.item.create(item={
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "input_text", "text": "\n\n".join(pending)}],
                        })
                        await connection.response.create(response=script_response(pending))
                    elif memos:
                        await send_memos(connection, memos)
                    elif confirm:
                        await connection.response.create(response=CONFIRM)
                    if state.get("announced") and mic is None and not carry:
                        mic = asyncio.create_task(listen_after(ear, speaker, connection, state))
                elif kind in ("response.output_audio.delta", "response.audio.delta") and not state["quiet"] and not state["drop_audio"]:
                    if state.get("needs_announce"):
                        state["got_audio"] = True
                    state["last_engage"] = time.monotonic()
                    speaker.write(base64.b64decode(event.delta))
                elif kind in ("response.output_audio_transcript.delta", "response.audio_transcript.delta") and event.delta and not state["quiet"]:
                    state["line"] += event.delta
                    if who != home["name"]:
                        print(f"\n{home['name']}: ", end="", flush=True)
                        who = home["name"]
                    print(event.delta, end="", flush=True)
                elif kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                    text = getattr(event, "transcript", "") or state["line"]
                    if not state["logged_line"]:
                        follow(state, "assistant", remember(log, "assistant", text))
                        state["logged_line"] = True
                    state["line"] = ""
                elif kind == "conversation.item.input_audio_transcription.updated":
                    text = getattr(event, "transcript", "") or ""
                    if show_person(state, text):
                        who = "person"
                    if text:
                        state["heard"] = text
                elif kind == "conversation.item.input_audio_transcription.completed":
                    text = getattr(event, "transcript", "") or state.get("heard") or ""
                    if state.get("naming") is None and voices is not None and not state.get("hold_reply"):
                        begin_name(voices, ear, state, connection)
                    await settle_speaker(state, voices)
                    if show_person(state, text):
                        who = "person"
                    note_heard(log, state, text)
                elif kind == "response.function_call_arguments.done" and event.call_id not in seen:
                    seen.add(event.call_id)
                    asyncio.create_task(finish_call(connection, speaker, state, event.name, event.call_id, event.arguments))
                elif kind == "error":
                    message = getattr(getattr(event, "error", None), "message", "") or ""
                    if "active response in progress" in message:
                        async with state["lock"]:
                            state["responding"] = True
                            if state.get("answering"):
                                state["redo_speaker"] = True
                            elif state.get("notes"):
                                pass
                            elif state["reminder_text"]:
                                state["later"] = state["reminder_text"]
                            else:
                                state["confirm"] = True
                        continue
                    print(f"\n{event}", flush=True)
                    record("error", message or str(event))
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
        async with AsyncOpenAI(api_key=os.environ["XAI_API_KEY"], base_url=XAI_BASE) as client:
            bank = Bank(DATA / "memories", home["memory"], client)
            voices = Voices(DATA / "voices", home["people"], home["voice_match"], home["voice_prior"])
            voices.load()
            saves = []
            try:
                await bank.load()
            except Exception as exc:
                record("error", f"memory: {exc}")
                status(f"memory unavailable ({exc})")
                bank = None
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
                audio = ear.engage(keep=kind != "timer", pause_seconds=home["pause_seconds"])
                begin()
                if reminder:
                    status(f"reminder due: {spoken_line(reminder)}")
                else:
                    status(f"heard {home['wake']}")
                speaker.write(bloop(True))
                if reminder:
                    status("opening session for a reminder")
                else:
                    status(f"opening session, {len(audio) / (RATE * 2):.1f}s buffered")
                said = True
                trace = Trace(time.strftime("%Y-%m-%d %H:%M"), home["name"])
                try:
                    reason, said = await run_session(
                        client, ear, speaker, started, current, home, tools, book, log, reminder, bank, trace, voices,
                    )
                except Exception as exc:
                    reason = str(exc).strip() or type(exc).__name__
                    said = False
                    record("error", reason)
                if trace.lines:
                    if bank is not None:
                        saves.append(asyncio.create_task(bank.keep(trace)))
                    if home["people"]:
                        saves.append(asyncio.create_task(revise(
                            client, home["memory"]["think_model"], trace, DATA / "people", home["people"],
                        )))
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
            if saves:
                await asyncio.gather(*saves, return_exceptions=True)
    finally:
        if ear is not None:
            ear.close()
        speaker.close()

if __name__ == "__main__":
    asyncio.run(main())
