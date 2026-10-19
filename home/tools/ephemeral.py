"""One-off Python. Saved, then run immediately."""

import asyncio
import inspect
import io
import sys
import time
from pathlib import Path

from ..log import record
from .maker import LIMIT, defines_call, device_api, ident, load_module

DIR = Path(__file__).resolve().parents[1] / "ephemeral"
SCRIPT = "Imports on their own lines at the top, then the steps. Top-level await is allowed. Do not define a function. print is returned to you as output.\n"
LIMIT_OUT = 8000

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        for stream in self.streams:
            stream.flush()

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

    def handles(self, name):
        return name == "run_script"

    def instructions(self, _home):
        return "\n".join([
            "When they ask for something the current tools do not do, run_script is the default.",
            "A concrete one-off, such as blink the lights, fetch something once, or wait and then do one thing, is a script. Write it and run it now.",
            "Do not make a lasting tool for that, and do not file an issue unless they explicitly say to file an issue or a bug.",
            "A lasting ability they will ask for again is save_tool, after you have talked through what it should do.",
            "Tell them you are writing code and what it will do. Do not hide that.",
            "run_script returns immediately with started. The conversation keeps going. A sleep does not block it.",
            "When the script finishes, a later message has output, which is everything it printed, or error.",
            "That output is the answer. Do not say you cannot see the machine.",
            "If error is a mistake in the code, fix it and call run_script again. Keep going until it works. Do not ask them to run it.",
            "Put imports on their own lines, then the steps. Top-level await is allowed. Do not define a function. Do not import home.",
            device_api(),
        ])

    def schemas(self, _home):
        return [{
            "type": "function",
            "name": "run_script",
            "description": "Write Python and run it on this computer now. Returns started immediately. A later message brings output (what print wrote) or error. If error is your code, fix it and call again. Tell the person you are writing code. A sleep does not block the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A few words naming this one run, such as blink the lights."},
                    "code": {"type": "string", "description": SCRIPT + device_api()},
                },
                "required": ["code"],
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

    async def call(self, _name, args, home):
        code = str(args.get("code") or "").strip()
        if not code:
            return {"error": "need the Python code"}
        if len(code) > LIMIT:
            return {"error": "that code is too long"}
        path = self._path(args.get("title"))
        runnable = prepare(code)
        path.write_text(runnable)
        print(f"\nscript {path.name}", flush=True)
        if self.deliver is None:
            return await self._execute(path, home)
        if self.track is not None:
            self.track["scripts"] = self.track.get("scripts", 0) + 1
        asyncio.create_task(self._background(path, home))
        return {
            "started": True,
            "file": path.name,
            "note": "Tell them you are writing code and what it will do. output or error arrives in a later message. If that error is your code, fix it and call run_script again.",
        }

    async def _background(self, path, home):
        try:
            payload = await self._execute(path, home)
        finally:
            if self.track is not None:
                self.track["scripts"] = max(0, self.track.get("scripts", 1) - 1)
        if self.deliver is not None:
            await self.deliver(finished_text(payload))

    async def _execute(self, path, home):
        captured = io.StringIO()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, captured), Tee(stderr, captured)
        try:
            module_name = "e_" + path.stem.replace("-", "_")
            module = load_module(module_name, path)
            result = module.call({}, home)
            if inspect.isawaitable(result):
                result = await result
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
            sys.stdout, sys.stderr = stdout, stderr
        payload = with_output(result, captured.getvalue())
        if payload.get("output"):
            record("script", f"{path.name}: {payload['output']}")
        return payload
