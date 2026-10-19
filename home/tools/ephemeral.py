"""One-off Python. Saved, then run immediately."""

import inspect
import time
from pathlib import Path

from .maker import LIMIT, defines_call, device_api, ident, load_module

DIR = Path(__file__).resolve().parents[1] / "ephemeral"

class Ephemeral:
    def __init__(self, directory=DIR):
        self.directory = Path(directory)

    def handles(self, name):
        return name == "run_script"

    def instructions(self, _home):
        return "\n".join([
            "When they ask for something the current tools do not do, run_script is the default.",
            "A concrete one-off, such as blink the lights, fetch something once, or wait and then do one thing, is a script. Write it and run it now.",
            "Do not make a lasting tool for that, and do not file an issue unless they explicitly say to file an issue or a bug.",
            "A lasting ability they will ask for again is save_tool, after you have talked through what it should do.",
            device_api(),
        ])

    def schemas(self, _home):
        return [{
            "type": "function",
            "name": "run_script",
            "description": "Write a one-off Python script and run it now. Not a lasting tool, and not an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A few words naming this one run, such as blink the lights."},
                    "code": {"type": "string", "description": device_api()},
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
        path.write_text(code)
        print(f"\nscript {path.name}", flush=True)
        try:
            if not defines_call(code):
                raise ValueError("define async def call(args, home)")
            module_name = "e_" + path.stem.replace("-", "_")
            module = load_module(module_name, path)
            result = module.call({}, home)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            return {"error": text}
        if isinstance(result, dict):
            return result
        return {"result": result}
