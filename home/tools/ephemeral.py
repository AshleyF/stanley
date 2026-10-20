"""One-off Python. Saved, then run immediately."""

import ast
import asyncio
import contextvars
import importlib.util
import inspect
import io
import re
import sys
import threading
import time
from pathlib import Path

from ..config import DATA
from ..log import record

DIR = DATA / "ephemeral"
SCRIPT = "Imports on their own lines at the top, then the steps. Top-level await is allowed. Do not define a function. print is returned to you as output.\n"
LIMIT = 40000
LIMIT_OUT = 8000
LIVE_DELAY = 0.3

def device_api():
    return "\n".join([
        "This runs on this computer, inside Home. The working directory is the home package.",
        "home is a dict already in scope. It is not a module. Do not import it.",
        "Do not invent imports. You may import asyncio, pathlib, json, and urllib.",
        "A light is: from home.tools.lights import Lights",
        "await Lights().call(\"set_light\", {\"name\": \"<a light name>\", \"on\": True, \"hue\": 0, \"saturation\": 100, \"brightness\": 100}, home)",
        "home is the last argument. Leaving it out fails. on false turns the light off. hue 0 is red and 120 is green. saturation and brightness are 0 to 100. Use a name from the light list.",
        "A shower is: from home.tools.shower import Shower",
        "await Shower().call(\"set_shower\", {\"name\": \"<a shower name>\", \"action\": \"on\", \"preset\": \"<a preset>\"}, home)",
        "action is on, off, or temperature. temperature is Fahrenheit, at or below that shower's maximum. The last argument is home.",
        "print() is the answer. That text comes back to you as output. A wait is await asyncio.sleep(seconds). The conversation continues while it sleeps.",
    ])

def ident(value):
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").casefold()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        return None
    return cleaned[:40]

def defines_call(code):
    tree = ast.parse(code)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "call"
        for node in tree.body
    )

def load_module(script_id, path):
    module_name = f"home_script_{script_id}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "call", None)):
        raise TypeError("the code must define call(args, home)")
    return module

class _Router:
    def __init__(self, slot, real):
        self.slot, self.real = slot, real
        self.var = None

    def _streams(self):
        current = self.var.get()
        if not current or not current.get("on"):
            return (self.real(),)
        return current["out"] if self.slot == 0 else current["err"]

    def write(self, data):
        streams = [stream for stream in self._streams() if stream is not None]
        for stream in streams:
            stream.write(data)
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        for stream in self._streams():
            if stream is not None:
                stream.flush()

class Capture:
    """Per-script stdout. Two scripts can run at once without mixing their output."""

    def __init__(self):
        self.var = contextvars.ContextVar("ephemeral_capture", default=None)
        self.depth = 0
        self.real_out = None
        self.real_err = None
        self.router_out = _Router(0, lambda: self.real_out)
        self.router_err = _Router(1, lambda: self.real_err)
        self.router_out.var = self.var
        self.router_err.var = self.var

    def begin(self, extras):
        if self.depth == 0:
            self.real_out, self.real_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = self.router_out, self.router_err
        self.depth += 1
        box = {"on": True, "out": [self.real_out, *extras], "err": [self.real_err, *extras]}
        return self.var.set(box)

    def end(self, token):
        current = self.var.get()
        if current:
            current["on"] = False
        self.var.reset(token)
        self.depth -= 1
        if self.depth == 0 and self.real_out is not None:
            sys.stdout, sys.stderr = self.real_out, self.real_err

CAPTURE = Capture()

class Live:
    def __init__(self, loop, push):
        self.loop, self.push = loop, push
        self.buf, self.waiting, self.alive = [], False, True
        self.lock = threading.Lock()

    def write(self, data):
        if not data or not self.alive:
            return 0
        with self.lock:
            if not self.alive:
                return 0
            self.buf.append(data)
            if not self.waiting:
                self.waiting = True
                self.loop.call_soon_threadsafe(self._arm, context=contextvars.Context())
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        return None

    def close(self):
        with self.lock:
            self.alive = False
            self.buf = []
            self.waiting = False

    def _arm(self):
        self.loop.call_later(LIVE_DELAY, self._flush)

    def _flush(self):
        with self.lock:
            if not self.alive:
                self.buf = []
                self.waiting = False
                return
            text = clip("".join(self.buf))
            self.buf = []
            self.waiting = False
        if text:
            asyncio.create_task(self._send(text))

    async def _send(self, text):
        if not self.alive:
            return
        try:
            await self.push(text)
        except Exception:
            pass

def strip_fences(code):
    text = code.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def prepare(code):
    code = strip_fences(code)
    try:
        if defines_call(code):
            return code
    except SyntaxError:
        pass
    head, body, started = [], [], False
    for line in code.splitlines():
        stripped = line.strip()
        if not started and (not stripped or stripped.startswith("#") or stripped.startswith("import ") or stripped.startswith("from ")):
            head.append(line)
            continue
        started = True
        body.append(line)
    if not any(line.strip() for line in body):
        body = ["return {}"]
    indented = "\n".join(("    " + line) if line.strip() else "" for line in body)
    return "\n".join(head + ["async def call(args, home):", indented]) + "\n"

def clip(text):
    text = (text or "").strip()
    if len(text) <= LIMIT_OUT:
        return text
    return text[:LIMIT_OUT] + "\n..."

def with_output(result, printed):
    printed = clip(printed)
    if result is None:
        payload = {"ok": True}
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"result": result}
    if printed:
        payload["output"] = printed
    return payload

def finished_text(payload):
    if payload.get("stopped"):
        lines = [f"Script {payload.get('id')} was stopped."]
        if payload.get("output"):
            lines.append("output:\n" + payload["output"])
        return "\n".join(lines)
    lines = ["A script you started has finished."]
    if payload.get("error"):
        lines.append("error: " + str(payload["error"]))
    if payload.get("output"):
        lines.append("output:\n" + payload["output"])
    elif not payload.get("error"):
        lines.append("It finished with no printed output.")
    return "\n".join(lines)

class Ephemeral:
    def __init__(self, directory=DIR):
        self.directory = Path(directory)
        self.deliver = None
        self.track = None
        self.keep = None
        self.running = {}

    def handles(self, name):
        return name in ("run_script", "stop_script")

    def instructions(self, _home):
        return "\n".join([
            "When they ask for something the current tools do not do, run_script is how you do it.",
            "Code is ephemeral. Write it and run it now. A light, a fetch, a wait, or anything else a script can do is a script.",
            "There is no tool to save for later. Do not talk about adding one.",
            "A deeper feature of Home, something that should be built in rather than run as a script, is an issue. Do not write a script for that.",
            "Tell them you are writing code and what it will do. Do not hide that.",
            "run_script returns immediately with started and an id. The conversation keeps going. A sleep does not block it.",
            "While it runs, later messages show what it prints. That output is for you to remember. Do not talk about it as it arrives.",
            "Speak during a run only for an error or something extraordinary, in one short sentence. If they ask what is going on, answer from what it printed.",
            "When it finishes, a message says so and includes output or error. output is the answer. Tell them that. Do not say you cannot see the machine.",
            "stop_script stops a running script. Pass its id. If only one is running, the id can be omitted. Stop one when they want it stopped.",
            "If error is a mistake in the code, fix it and call run_script again. Keep going until it works. Do not ask them to run it.",
            "Put imports on their own lines, then the steps. Top-level await is allowed. Do not define a function. Do not import home.",
            device_api(),
        ])

    def schemas(self, _home):
        return [{
            "type": "function",
            "name": "run_script",
            "description": "Write Python and run it on this computer now. Returns started and an id immediately. Printed lines arrive while it runs. A finished message brings the full output or error. If error is your code, fix it and call again. Tell the person you are writing code. A sleep does not block the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A few words naming this one run, such as blink the lights."},
                    "code": {"type": "string", "description": SCRIPT + device_api()},
                },
                "required": ["code"],
            },
        }, {
            "type": "function",
            "name": "stop_script",
            "description": "Stop a script that is still running. Pass the id from started. Omit id if only one is running.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The script id. The file name from started."}},
            },
        }]

    def _path(self, title):
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = ident(title) or "script"
        path = self.directory / f"{stamp}-{slug}.py"
        if path.exists():
            path = self.directory / f"{stamp}-{slug}-{time.time_ns() % 10000:04d}.py"
        return path

    async def call(self, name, args, home):
        if name == "stop_script":
            return self.stop(args.get("id"))
        code = str(args.get("code") or "").strip()
        if not code:
            return {"error": "need the Python code"}
        if len(code) > LIMIT:
            return {"error": "that code is too long"}
        path = self._path(args.get("title"))
        runnable = prepare(code)
        path.write_text(runnable)
        if self.keep is not None:
            self.keep(path.name, runnable)
        print(f"\nscript {path.name}", flush=True)
        if self.deliver is None:
            return await self._execute(path, home)
        script_id = path.name
        if self.track is not None:
            self.track["scripts"] = self.track.get("scripts", 0) + 1
        task = asyncio.create_task(self._background(path, home, script_id))
        self.running[script_id] = task
        return {
            "started": True,
            "id": script_id,
            "note": "Tell them you are writing code and what it will do. Printed lines arrive while it runs. stop_script with this id stops it.",
        }

    def stop(self, script_id):
        wanted = str(script_id or "").strip()
        if not self.running:
            return {"error": "no script is running"}
        if not wanted:
            if len(self.running) == 1:
                wanted = next(iter(self.running))
            else:
                return {"error": "which script?", "running": list(self.running)}
        task = self.running.get(wanted)
        if task is None:
            hits = [key for key in self.running if wanted in key]
            if len(hits) == 1:
                wanted, task = hits[0], self.running[hits[0]]
        if task is None:
            return {"error": "that script is not running", "running": list(self.running)}
        task.cancel()
        return {"stopping": wanted}

    async def _background(self, path, home, script_id):
        payload = {"stopped": True, "id": script_id}
        try:
            payload = await self._execute(path, home, script_id)
        except asyncio.CancelledError:
            pass
        finally:
            self.running.pop(script_id, None)
            if self.track is not None:
                self.track["scripts"] = max(0, self.track.get("scripts", 1) - 1)
        if self.deliver is None:
            return
        task = asyncio.current_task()
        if task is not None and getattr(task, "uncancel", None):
            while task.cancelling():
                task.uncancel()
        await self.deliver(finished_text(payload))

    async def _execute(self, path, home, script_id=None):
        captured = io.StringIO()
        live = None
        extras = [captured]
        if self.deliver is not None and script_id:
            live = Live(asyncio.get_running_loop(), lambda text: self.deliver(
                f"Script {script_id} is still running. It printed:\n{text}"
            ))
            extras.append(live)
        token = CAPTURE.begin(extras)
        try:
            try:
                module_name = "e_" + path.stem.replace("-", "_")
                module = load_module(module_name, path)
                result = module.call({}, home)
                if inspect.isawaitable(result):
                    result = await result
            except asyncio.CancelledError:
                payload = {"stopped": True, "id": script_id}
                printed = clip(captured.getvalue())
                if printed:
                    payload["output"] = printed
                return payload
            except Exception as exc:
                text = str(exc).strip() or type(exc).__name__
                printed = clip(captured.getvalue())
                payload = {"error": text}
                if printed:
                    payload["output"] = printed
                print(f"\nscript failed — {text}", flush=True)
                record("script", f"failed {path.name}: {text}")
                return payload
        finally:
            if live is not None:
                live.close()
            CAPTURE.end(token)
        payload = with_output(result, captured.getvalue())
        if payload.get("output"):
            record("script", f"{path.name}: {payload['output']}")
        return payload
