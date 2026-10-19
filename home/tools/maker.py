"""Tools Stanley writes for himself. Saved on disk and registered immediately."""

import ast
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "dynamic"
MANIFEST = DIR / "manifest.json"
LIMIT = 40000

def device_api():
    return "\n".join([
        "Code shape: async def call(args, home). args is a dict. home is the house config. Return a dict.",
        "Lights: from home.tools.lights import Lights",
        "await Lights().call(\"set_light\", {\"name\": \"<a light name>\", \"on\": True, \"hue\": 120, \"saturation\": 100, \"brightness\": 80}, home)",
        "on false turns it off. hue is 0 to 360, and green is 120. saturation and brightness are 0 to 100. Use a name from the light list.",
        "Shower: from home.tools.shower import Shower",
        "await Shower().call(\"set_shower\", {\"name\": \"<a shower name>\", \"action\": \"on\", \"preset\": \"<a preset>\"}, home)",
        "action is on, off, or temperature. temperature is Fahrenheit, at or below that shower's maximum. preset is optional. Use names from the shower list.",
        "A reminder only speaks later. It does not run code. To do something after a wait, await asyncio.sleep(seconds) and then do the work.",
        "For anything else, write ordinary Python. The standard library, including urllib, is available.",
    ])

def ident(value):
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").casefold()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        return None
    return cleaned[:40]

def parameters_of(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("parameters must be a JSON object")
    properties = value.get("properties") if isinstance(value.get("properties"), dict) else {}
    required = value.get("required") if isinstance(value.get("required"), list) else []
    required = [str(item) for item in required]
    return {"type": "object", "properties": properties, "required": required}

def defines_call(code):
    tree = ast.parse(code)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "call"
        for node in tree.body
    )

def load_module(tool_id, path):
    module_name = f"home_dynamic_{tool_id}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "call", None)):
        raise TypeError("the code must define call(args, home)")
    return module

class DynamicTool:
    def __init__(self, entry, module, code):
        self.entry, self.module, self.code = entry, module, code

    def handles(self, name):
        return self.module is not None and name == self.entry["name"]

    def instructions(self, _home):
        return self.entry.get("instructions") or ""

    def schemas(self, _home):
        if self.module is None:
            return []
        return [{
            "type": "function",
            "name": self.entry["name"],
            "description": self.entry.get("description") or self.entry["name"],
            "parameters": self.entry.get("parameters") or {"type": "object", "properties": {}},
        }]

    async def call(self, _name, args, home):
        if self.module is None:
            return {"error": self.entry.get("error") or "this tool did not load"}
        try:
            result = self.module.call(args, home)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            return {"error": text}
        if isinstance(result, dict):
            return result
        return {"result": result}

class Shelf:
    def __init__(self, directory=DIR):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.tools = []
        self.bucket = None
        self.load()

    def attach(self, bucket):
        self.bucket = bucket
        for tool in self.tools:
            bucket.append(tool)

    def load(self):
        self.tools = []
        for entry in self._read():
            path = self.directory / f"{entry['id']}.py"
            code, module, error = "", None, ""
            try:
                code = path.read_text()
                module = load_module(entry["id"], path)
            except (OSError, SyntaxError, Exception) as exc:
                error = str(exc).strip() or type(exc).__name__
            if error:
                entry["error"] = error
            self.tools.append(DynamicTool(entry, module, code))

    def _read(self):
        try:
            data = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        found = []
        for entry in data.get("tools") or []:
            if ident(entry.get("id")) and ident(entry.get("name")):
                found.append(dict(entry))
        return found

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        body = {"tools": [tool.entry for tool in self.tools]}
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(self.manifest_path)

    def summary(self):
        if not self.tools:
            return "You have no added tools yet."
        lines = []
        for tool in self.tools:
            broken = " (not loaded)" if tool.module is None else ""
            lines.append(f"- {tool.entry['id']} calls {tool.entry['name']}: {tool.entry.get('description') or ''}{broken}")
        return "Added tools:\n" + "\n".join(lines)

    def names_used(self, home, except_id=None):
        names = {"list_tools", "save_tool", "delete_tool", "end_session", "not_for_me"}
        for tool in self.bucket or []:
            if isinstance(tool, DynamicTool):
                if tool.entry["id"] != except_id:
                    names.add(tool.entry["name"])
                continue
            if isinstance(tool, Maker):
                continue
            try:
                schemas = tool.schemas(home)
            except Exception:
                continue
            names.update(schema.get("name") for schema in schemas if schema.get("name"))
        return names

    def get(self, tool_id):
        return next((tool for tool in self.tools if tool.entry["id"] == tool_id), None)

    def list(self, tool_id=None):
        if tool_id:
            wanted = ident(tool_id)
            tool = self.get(wanted) if wanted else None
            if tool is None:
                return {"error": "no tool with that id", "tools": [item.entry["id"] for item in self.tools]}
            entry = dict(tool.entry)
            entry["code"] = tool.code
            return {"tool": entry}
        return {"tools": [
            {"id": tool.entry["id"], "name": tool.entry["name"], "description": tool.entry.get("description") or ""}
            for tool in self.tools
        ]}

    def save(self, home, tool_id, name, description, parameters, code, instructions):
        tool_id, name = ident(tool_id), ident(name)
        description = " ".join(str(description or "").split())
        instructions = str(instructions or "").strip() or description
        code = str(code or "").strip()
        if not tool_id or not name:
            return {"error": "id and name must start with a letter, then letters, digits, or underscores"}
        if not description or not code:
            return {"error": "need a description and the Python code"}
        if len(code) > LIMIT:
            return {"error": "that code is too long"}
        if name in self.names_used(home, except_id=tool_id):
            return {"error": f"{name} is already a tool. Pick another name, or update that id."}
        try:
            parameters = parameters_of(parameters)
            if not defines_call(code):
                raise ValueError("the code must define call(args, home) at the top")
        except (ValueError, json.JSONDecodeError, SyntaxError) as exc:
            return {"error": str(exc).strip() or type(exc).__name__}
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{tool_id}.py"
        previous = path.read_text() if path.is_file() else None
        path.write_text(code)
        try:
            module = load_module(tool_id, path)
        except Exception as exc:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous)
            return {"error": str(exc).strip() or type(exc).__name__}
        entry = {
            "id": tool_id,
            "name": name,
            "description": description,
            "instructions": instructions,
            "parameters": parameters,
        }
        current = self.get(tool_id)
        if current is None:
            tool = DynamicTool(entry, module, code)
            self.tools.append(tool)
            if self.bucket is not None:
                self.bucket.append(tool)
        else:
            current.entry, current.module, current.code = entry, module, code
        self._write()
        print(f"\ntool {tool_id} saved", flush=True)
        return {"saved": tool_id, "name": name, "refresh_tools": True}

    def delete(self, tool_id):
        wanted = ident(tool_id)
        tool = self.get(wanted) if wanted else None
        if tool is None:
            return {"error": "no tool with that id", "tools": [item.entry["id"] for item in self.tools]}
        self.tools.remove(tool)
        if self.bucket is not None and tool in self.bucket:
            self.bucket.remove(tool)
        self._write()
        (self.directory / f"{tool.entry['id']}.py").unlink(missing_ok=True)
        print(f"\ntool {tool.entry['id']} deleted", flush=True)
        return {"deleted": tool.entry["id"], "refresh_tools": True}

class Maker:
    def __init__(self, shelf):
        self.shelf = shelf

    def handles(self, name):
        return name in ("list_tools", "save_tool", "delete_tool")

    def instructions(self, _home):
        return "\n".join([
            "You can add, change, and remove lasting tools. They are saved and can be called immediately, including in this session.",
            "A concrete one-off is run_script, not save_tool. Blink the lights, fetch something once, or wait and then do one thing.",
            "save_tool is for an ability they will ask for again. Before you save one, talk until you know what it should do, when it should happen, and what done looks like.",
            "Do not file an issue unless they explicitly say to file an issue or a bug.",
            "Call devices with the same code run_script uses.",
            "list_tools shows the id and name of each added tool. Pass an id to read that tool, including its code, before you change it.",
            "save_tool creates a tool when the id is new and replaces it when the id already exists.",
            "delete_tool removes one by id.",
            self.shelf.summary(),
        ])

    def schemas(self, _home):
        code = {
            "type": "string",
            "description": device_api(),
        }
        return [
            {
                "type": "function",
                "name": "list_tools",
                "description": "List the tools you added. Pass an id to read one, including its code and parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "The tool id. Omit it to list them all."}},
                },
            },
            {
                "type": "function",
                "name": "save_tool",
                "description": "Create or replace a lasting tool they will ask for again. Not for a one-off; that is run_script. Talk first. A new id creates it. An existing id replaces it. Do not use this instead of an explicit request to file an issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Stable id, such as green_later. Letters, digits, and underscores."},
                        "name": {"type": "string", "description": "The function name you will call, such as turn_light_green_later."},
                        "description": {"type": "string", "description": "What the tool does, for you, when you decide to call it."},
                        "instructions": {"type": "string", "description": "When to call it. Omit this to use the description."},
                        "parameters": {"type": "object", "description": "JSON schema for the arguments: type object, properties, required."},
                        "code": code,
                    },
                    "required": ["id", "name", "description", "parameters", "code"],
                },
            },
            {
                "type": "function",
                "name": "delete_tool",
                "description": "Delete one added tool by its id. Built-in tools cannot be deleted.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        ]

    async def call(self, name, args, home):
        if name == "list_tools":
            return self.shelf.list(args.get("id"))
        if name == "save_tool":
            return self.shelf.save(
                home,
                args.get("id"),
                args.get("name"),
                args.get("description"),
                args.get("parameters"),
                args.get("code"),
                args.get("instructions"),
            )
        if name == "delete_tool":
            return self.shelf.delete(args.get("id"))
        return {"error": f"unknown tool {name}"}
