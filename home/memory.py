"""Flat-file memory. Text notes beside their embedding vectors.

home/data/memories/<id>.txt is the note. <id>.mem is the vector, with the
vector's length cached beside it. A model name with a slash is a local
Hugging Face model. Any other name is sent to the xAI embeddings API.
Home rebuilds the search index from those files when it starts. Edit or
add a text file and the next start embeds it. Delete a text file and the
leftover vector is removed.
"""

import hashlib
import json
import re
import struct
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .config import DATA
from .hnsw import HNSW
from .log import record

MAGIC = b"SMEM"
FROM = re.compile(r"^From:\s*(.+)$")
DIR = DATA / "memories"
# Unrelated sentences. Their average direction is the cone every embedding
# points along. That part is removed before a comparison, so cosine is the
# leftover difference instead of "both are sentences."
PROBES = (
    "The river froze overnight.",
    "Please pass the salt.",
    "A bicycle has two wheels.",
    "The meeting starts at noon.",
    "Copper conducts electricity.",
    "She closed the window.",
    "The book is on the table.",
    "Add one cup of flour.",
    "The train leaves at six.",
    "Birds migrate in the fall.",
    "He locked the front door.",
    "The paint is still wet.",
    "Two plus two is four.",
    "The cat slept on the chair.",
    "Clouds covered the moon.",
    "Turn left at the corner.",
    "The soup needs more salt.",
    "Paper burns at a low temperature.",
    "The keys are in the drawer.",
    "Rain fell all morning.",
)

SPLIT = """You turn one conversation into memories. Return JSON only, no markdown: {"memories": ["...", "..."]}.
Each string is one memory about one topic. Several topics in one conversation become several memories. A person's name, a birthday, and a hobby are different memories. Skip talk that has nothing worth keeping.
A memory is not the transcript. Keep what would matter later.
When a quotation is the part worth remembering, keep those words verbatim.
When code was written, keep the code that mattered, verbatim, especially if the person said it worked. Code is a better record than a description of the same thing. Drop code that failed and was replaced, unless that failure itself matters.
Start every memory with a line "From: <when>" using the session time you are given. Nothing comes before that line."""

MERGE = """You merge memories that are about the same thing into one. Return JSON only, no markdown: {"memory": "..."}.
Keep whatever is new in any of them. Drop repeated overlap. Keep the differences.
Keep quotations and code verbatim when they are the part worth keeping.
The opening lines are provenance. Keep every "From: ..." line from the inputs, one per line, no duplicates, and nothing before them."""

JUDGE = """You decide whether a memory answers what was just said. Return JSON only, no markdown: {"keep": [0]}.
The person speaking is the resident the memories are about. "My" means that resident. A spouse named in a memory is not the speaker.
A question about a fact is answered by the memory that states that fact. "Do you remember my birthday?" is answered by the resident's own birthday.
Return {"keep": []} for a greeting or small talk, and for a memory that only shares a name."""

def recall_note(memos):
    body = "\n\n".join(memos)
    return (
        "Speak now. A memory was just handed to you because it answers what they asked. "
        "You may have already said you did not know. Correct that out loud. "
        "Say you just remembered, then the fact, in one or two short sentences. "
        "Do not say you do not know. Do not stay silent. Do not call a tool. "
        "Use only the fact below.\n\n"
        + body
    )


def label(text):
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if line and not line.startswith("From:"):
            return line[:42]
    return "note"


def note(text):
    print(f"\n[{text}]", flush=True)
    record("memory", text)


_LOCAL = {}
_LOCAL_LOCK = threading.Lock()


def _local_vector(model_name, text):
    with _LOCAL_LOCK:
        loaded = _LOCAL.get(model_name)
        if loaded is None:
            note(f"memory: loading {model_name}")
            import os
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import torch
            from transformers import AutoModel, AutoTokenizer
            try:
                from transformers.utils.logging import disable_progress_bar
                disable_progress_bar()
            except ImportError:
                pass
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            loaded = (tokenizer, model, torch)
            _LOCAL[model_name] = loaded
        tokenizer, model, torch = loaded
        batch = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, 0.0)
        counts = batch["attention_mask"].sum(1).clamp(min=1).unsqueeze(-1)
        return (hidden.sum(1) / counts)[0].numpy()


def action_line(name, result):
    if not isinstance(result, dict):
        return f"{name}: {result}"
    if result.get("error"):
        return f"{name} failed: {result['error']}"
    skip = {"note"}
    bits = []
    for key, value in result.items():
        if key in skip or value in (None, "", [], {}):
            continue
        bits.append(f"{key} {value}")
    return f"{name}: " + ", ".join(bits) if bits else name


class Trace:
    """What happened in one session, for the model that writes memories."""

    def __init__(self, when, agent):
        self.when, self.agent, self.lines = when, agent, []

    def say(self, who, text):
        text = " ".join((text or "").split())
        if text:
            self.lines.append(f"{who}: {text}")

    def did(self, text):
        text = " ".join((text or "").split())
        if text:
            self.lines.append(f"Did: {text}")

    def code(self, title, source):
        source = (source or "").strip()
        if source:
            self.lines.append(f"Wrote {title}:\n{source}")

    def dossier(self):
        if not self.lines:
            return ""
        body = "\n\n".join(self.lines)
        if len(body) > 80000:
            body = body[:80000] + "\n..."
        return f"Session: {self.when}\nAgent: {self.agent}\n\n{body}"


def from_stamps(text):
    found = []
    for line in (text or "").splitlines():
        match = FROM.match(line.strip())
        if match and match.group(1).strip():
            found.append(match.group(1).strip())
    return found


def with_from(text, stamps):
    kept = []
    for stamp in [*from_stamps(text), *stamps]:
        if stamp and stamp not in kept:
            kept.append(stamp)
    body = "\n".join(line for line in (text or "").splitlines() if not FROM.match(line.strip())).strip()
    head = "\n".join(f"From: {stamp}" for stamp in kept)
    if body:
        return f"{head}\n\n{body}\n" if head else f"{body}\n"
    return f"{head}\n" if head else ""


def _json(raw):
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def parse_texts(raw):
    data = _json(raw)
    if isinstance(data, str):
        return [data.strip()] if data.strip() else []
    if isinstance(data, dict):
        many = data.get("memories")
        if many is None and data.get("memory"):
            many = [data.get("memory")]
        data = many
    if not isinstance(data, list):
        return []
    found = []
    for item in data:
        if isinstance(item, str) and item.strip():
            found.append(item.strip())
        elif isinstance(item, dict):
            piece = item.get("text") or item.get("memory") or ""
            if str(piece).strip():
                found.append(str(piece).strip())
    return found


def parse_memory(raw):
    data = _json(raw)
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        piece = data.get("memory") or data.get("text") or ""
        if isinstance(piece, str) and piece.strip():
            return piece.strip()
    many = parse_texts(raw)
    return many[0] if many else ""


def digest(model, payload):
    hashed = hashlib.sha256()
    hashed.update(b"passage:v1\0")
    hashed.update(model.encode())
    hashed.update(b"\0")
    hashed.update(payload)
    return hashed.digest()


def pack_vector(vector, norm, hashed):
    values = np.asarray(vector, dtype=np.float32)
    header = struct.pack("<4sBIf32s", MAGIC, 1, values.size, float(norm), hashed)
    return header + values.tobytes()


def unpack_vector(blob):
    if len(blob) < 4 + 1 + 4 + 4 + 32:
        return None
    magic, version, dim, norm, hashed = struct.unpack_from("<4sBIf32s", blob)
    if magic != MAGIC or version != 1 or dim <= 0:
        return None
    start = struct.calcsize("<4sBIf32s")
    need = start + dim * 4
    if len(blob) < need:
        return None
    values = np.frombuffer(blob[start:need], dtype=np.float32).copy()
    if values.size != dim:
        return None
    return values, float(norm), hashed


class Bank:
    def __init__(self, directory, settings, client=None, embed=None, think=None):
        self.directory = Path(directory)
        self.embed_model = settings["embed_model"]
        self.think_model = settings["think_model"]
        self.recall_at = float(settings["recall"])
        self.merge_at = float(settings["merge"])
        self.client = client
        self._embed = embed
        self._think = think
        self.index = HNSW()
        self.text = {}
        self.lock = None
        self.cone = None

    def _guard(self):
        import asyncio
        if self.lock is None:
            self.lock = asyncio.Lock()
        return self.lock

    async def load(self):
        async with self._guard():
            self.directory.mkdir(parents=True, exist_ok=True)
            notes = {path.stem: path for path in self.directory.glob("*.txt")}
            vectors = {path.stem: path for path in self.directory.glob("*.mem")}
            for stem, path in vectors.items():
                if stem not in notes:
                    path.unlink(missing_ok=True)
                    note(f"dropped orphan embedding {path.name}")
            self.index = HNSW()
            self.text = {}
            fresh = 0
            for stem, path in sorted(notes.items()):
                payload = path.read_bytes()
                wanted = digest(self.embed_model, payload)
                vector = self._stored(vectors.get(stem), wanted)
                if vector is None:
                    try:
                        vector = await self.embed(payload.decode("utf-8", "replace"), "passage")
                    except Exception as exc:
                        record("memory", f"embed {path.name} failed: {exc}")
                        continue
                    self._write_vector(stem, vector, wanted)
                    fresh += 1
                body = payload.decode("utf-8", "replace")
                self.text[stem] = body
                self.index.add(stem, await self._focus(vector))
            kept = len(self.text)
            if fresh:
                note(f"memory: {kept} notes, embedded {fresh}")
            else:
                note(f"memory: {kept} notes")

    def _stored(self, path, wanted):
        if path is None or not path.is_file():
            return None
        try:
            unpacked = unpack_vector(path.read_bytes())
        except OSError:
            return None
        if unpacked is None:
            return None
        values, _norm, hashed = unpacked
        if hashed != wanted:
            return None
        return values

    def _write_vector(self, stem, vector, hashed):
        values = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(values)) or 1.0
        path = self.directory / f"{stem}.mem"
        tmp = path.with_suffix(".mem.tmp")
        tmp.write_bytes(pack_vector(values, norm, hashed))
        tmp.replace(path)

    def _write_text(self, stem, text):
        path = self.directory / f"{stem}.txt"
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(text)
        tmp.replace(path)
        return path.read_bytes()

    async def embed(self, text, kind):
        if self._embed is not None:
            return await self._embed(text, kind)
        prefixed = f"{kind}: {text}"
        if "/" in self.embed_model:
            import asyncio
            return await asyncio.to_thread(_local_vector, self.embed_model, prefixed)
        response = await self.client.embeddings.create(
            model=self.embed_model,
            input=[prefixed],
            encoding_format="float",
        )
        return response.data[0].embedding

    async def _focus(self, vector):
        if self.cone is None:
            units = []
            for sentence in PROBES:
                row = np.asarray(await self.embed(sentence, "passage"), dtype=np.float64)
                length = np.linalg.norm(row)
                if length:
                    units.append(row / length)
            mean = np.mean(units, axis=0)
            self.cone = mean / np.linalg.norm(mean)
        values = np.asarray(vector, dtype=np.float64)
        values = values - self.cone * float(values @ self.cone)
        return values.astype(np.float32)

    async def think(self, system, user):
        if self._think is not None:
            return await self._think(system, user)
        response = await self.client.chat.completions.create(
            model=self.think_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            reasoning_effort="low",
        )
        return response.choices[0].message.content or ""

    async def recall(self, text):
        text = " ".join((text or "").split())
        if len(text) < 12 or not self.text:
            return []
        vector = await self._focus(await self.embed(text, "query"))
        floor = max(0.0, self.recall_at - 0.06)
        async with self._guard():
            hits = self.index.search(vector, 3)
            near = []
            for key, score in hits:
                if key not in self.text or score < floor:
                    continue
                near.append({"id": key, "text": self.text[key], "score": score})
        found = [hit for hit in near if hit["score"] >= self.recall_at]
        if any(hit["score"] < self.recall_at for hit in near):
            try:
                found = await self._judge(text, near)
            except Exception as exc:
                record("memory", f"judge failed: {exc}")
        if found:
            ranks = ", ".join(f"{hit['score']:.2f} {label(hit['text'])}" for hit in found)
            short = text if len(text) <= 60 else text[:57] + "..."
            note(f"memory: placed {len(found)} for \"{short}\" ({ranks})")
        return found

    async def _judge(self, text, hits):
        lines = [f"Said: {text}", ""]
        for index, hit in enumerate(hits):
            lines.append(f"[{index}] {hit['text'].strip()}")
        raw = await self.think(JUDGE, "\n".join(lines))
        data = _json(raw) or {}
        keep = data.get("keep") if isinstance(data, dict) else None
        if not isinstance(keep, list):
            return []
        chosen = []
        for index in keep:
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(hits):
                chosen.append(hits[index])
        return chosen

    async def keep(self, trace):
        dossier = trace.dossier()
        if not dossier:
            return
        async with self._guard():
            try:
                raw = await self.think(SPLIT, f"Session time: {trace.when}\n\n{dossier}")
            except Exception as exc:
                record("memory", f"split failed: {exc}")
                return
            memories = [with_from(item, [trace.when]) for item in parse_texts(raw)]
            memories = [item for item in memories if item.strip()][:12]
            if not memories:
                return
            saved = 0
            for item in memories:
                try:
                    if await self._insert(item, [trace.when]):
                        saved += 1
                except Exception as exc:
                    record("memory", f"save failed: {exc}")
            if saved:
                note(f"memory: kept {saved}")

    async def _insert(self, text, stamps):
        text = with_from(text, stamps)
        vector = await self._focus(await self.embed(text, "passage"))
        hits = [(key, score) for key, score in self.index.search(vector, 8) if score >= self.merge_at and key in self.text]
        if hits:
            others = [self.text[key] for key, _score in hits]
            try:
                raw = await self.think(MERGE, "\n\n---\n\n".join([text, *others]))
                merged = parse_memory(raw)
            except Exception as exc:
                record("memory", f"merge failed: {exc}")
                merged = ""
            if merged:
                stamps = []
                for piece in [text, *others]:
                    for stamp in from_stamps(piece):
                        if stamp not in stamps:
                            stamps.append(stamp)
                text = with_from(merged, stamps)
                vector = await self._focus(await self.embed(text, "passage"))
                for key, _score in hits:
                    self._drop(key)
        stem = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        payload = self._write_text(stem, text)
        self._write_vector(stem, vector, digest(self.embed_model, payload))
        self.text[stem] = text
        self.index.add(stem, vector)
        return True

    def _drop(self, stem):
        self.index.discard(stem)
        self.text.pop(stem, None)
        for suffix in (".txt", ".mem"):
            path = self.directory / f"{stem}{suffix}"
            path.unlink(missing_ok=True)
