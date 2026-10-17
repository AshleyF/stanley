"""Speech in, speech out, for the living-room reading bulb at 192.168.4.38."""

import argparse, asyncio, contextlib, json, sys, tempfile
from pathlib import Path

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_ID, HOST, RATE_OUT = "Qwen/Qwen2.5-Omni-3B", "192.168.4.38", 24000
SYSTEM = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
COLORS = {"purple": (280, 100, 100), "red": (0, 100, 100), "blue": (240, 100, 100), "green": (120, 100, 100), "warm white": (30, 20, 100)}
COLOR_SCHEMA = {"type": "string", "enum": ["purple", "red", "blue", "green", "warm white"]}
TOOLS = (
    {"type": "function", "function": {"name": "lights_on", "description": "Turn the living-room reading lamp on.", "parameters": {"type": "object", "properties": {"color": COLOR_SCHEMA}}}},
    {"type": "function", "function": {"name": "lights_off", "description": "Turn the living-room reading lamp off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_light_color", "description": "Set the living-room reading lamp color and turn it on.", "parameters": {"type": "object", "properties": {"color": COLOR_SCHEMA}, "required": ["color"]}}},
)

def tool_block():
    """This checkpoint's chat template has no tools argument. Qwen's trained registration is this system text."""
    body = "\n".join(json.dumps(tool, separators=(",", ":")) for tool in TOOLS)
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{body}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )

def color_name(value):
    text = " ".join(str(value).split()).lower()
    if text not in COLORS:
        raise ValueError(f"unknown color: {text or value}")
    return text

def reported(exc):
    text = str(exc).strip() or type(exc).__name__
    print(text, flush=True)
    return {"error": text}

async def drive(action, color=None):
    from kasa import DeviceConfig, Module
    from kasa.iot import IotBulb
    bulb = None
    try:
        bulb = IotBulb(HOST, config=DeviceConfig(host=HOST, timeout=5))
        await bulb.update()
        if action == "off":
            await bulb.turn_off()
            print("lights off", flush=True)
            return {"on": False}
        await bulb.turn_on()
        if not color:
            print("lights on", flush=True)
            return {"on": True}
        name = color_name(color)
        hue, saturation, value = COLORS[name]
        await bulb.modules[Module.Light].set_hsv(hue, saturation, value)
        print(f"lights on, {name}", flush=True)
        return {"on": True, "color": name, "hsv": [hue, saturation, value]}
    except Exception as exc:
        return reported(exc)
    finally:
        if bulb is not None:
            with contextlib.suppress(Exception):
                await bulb.disconnect()

def run_tool(name, args):
    color = (args or {}).get("color")
    try:
        if name == "lights_off":
            return asyncio.run(drive("off"))
        if name not in ("lights_on", "set_light_color"):
            return {"error": f"unknown tool {name}"}
        if name == "set_light_color" and not color:
            print("color is required", flush=True)
            return {"error": "color is required"}
        if color:
            color_name(color)
        return asyncio.run(drive("on", color or None))
    except Exception as exc:
        return reported(exc)

def tool_calls(text):
    """JSON inside tool_call tags only. An example in prose must not move the lamp."""
    decoder, calls = json.JSONDecoder(), []
    for chunk in text.split("<tool_call>")[1:]:
        body = chunk.split("</tool_call>", 1)[0]
        start = body.find("{")
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            continue
        raw = (obj.get("arguments") or {}) if isinstance(obj, dict) else None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if isinstance(obj, dict) and obj.get("name") and isinstance(raw, dict):
            calls.append((obj["name"], raw))
    return calls

def device_index(kind):
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        key = "max_input_channels" if kind == "in" else "max_output_channels"
        return next((i for i, device in enumerate(devices) if device.get(key, 0) > 0), None)
    except Exception:
        return None

def load_wav(path):
    import numpy as np, soundfile as sf
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if rate != 16000 and len(data):
        count = max(1, int(round(len(data) * 16000 / rate)))
        data = np.interp(np.linspace(0, 1, count, endpoint=False), np.linspace(0, 1, len(data), endpoint=False), data)
    return np.ascontiguousarray(data, dtype=np.float32)

def complete(model, processor, wav, speak, earlier=None, extra=None):
    # Speech requires the exact system sentence. The text pass is tool registration only, and it must open with a tool call.
    system = SYSTEM if speak else tool_block()
    user = [{"type": "audio", "audio": load_wav(wav)}]
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": user}]
    if earlier:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": earlier}]})
    if extra:
        messages.append({"role": "user", "content": [{"type": "text", "text": extra}]})
    batch = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to("cpu", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
    if not speak:
        prefix = processor.tokenizer.encode("<tool_call>\n", add_special_tokens=False, return_tensors="pt").to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([batch["input_ids"], prefix], dim=-1)
        mask = batch.get("attention_mask")
        if torch.is_tensor(mask):
            batch["attention_mask"] = torch.cat([mask, torch.ones_like(prefix)], dim=-1)
    kwargs = {"thinker_do_sample": False, "thinker_max_new_tokens": 64 if speak else 96, "generation_mode": "audio" if speak else "text"}
    if speak:
        kwargs.update(speaker="Chelsie", talker_max_new_tokens=512)
    with torch.inference_mode():
        generated = model.generate(**batch, **kwargs)
    ids = generated[0] if isinstance(generated, tuple) else generated
    ids = getattr(ids, "sequences", ids)
    ids = ids.unsqueeze(0) if ids.ndim == 1 else ids
    n = batch["input_ids"].shape[-1]
    ids = ids[:, n:] if ids.shape[-1] > n else ids
    text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    if not speak:
        text = "<tool_call>\n" + text
    audio = generated[1] if isinstance(generated, tuple) and len(generated) > 1 else None
    audio = (audio[0] if audio else None) if isinstance(audio, (list, tuple)) else audio
    samples = None if audio is None else audio.reshape(-1).detach().float().cpu().numpy()
    return text, samples

def play_or_save(samples):
    import numpy as np, soundfile as sf
    samples = np.asarray(samples, dtype="float32").reshape(-1)
    device = device_index("out")
    if device is None:
        path = Path.cwd() / "reply.wav"
        sf.write(path, samples, RATE_OUT)
        print(f"wrote {path}", flush=True)
        return
    import sounddevice as sd
    sd.play(samples, RATE_OUT, device=device)
    sd.wait()
    print("played reply", flush=True)

def record_until_silence(rate=16000):
    """Mic until a short silence after speech. Not full duplex."""
    import numpy as np, sounddevice as sd, soundfile as sf
    frame, quiet_need, speech = rate // 20, 12, 500
    blocks, heard, quiet = [], False, 0
    print("Listening until a short silence. Ctrl-C stops.", flush=True)
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=frame, device=device_index("in")) as stream:
        while len(blocks) < 20 * 15:
            block, _ = stream.read(frame)
            blocks.append(block.copy())
            loud = float(np.abs(block).mean()) > speech
            heard, quiet = heard or loud, 0 if loud or not heard else quiet + 1
            if (heard and quiet >= quiet_need) or (not heard and len(blocks) >= 20 * 8):
                break
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    path = Path(tmp.name)
    sf.write(path, np.concatenate(blocks), rate)
    print(f"Recorded {path}", flush=True)
    return path

def one_turn(model, processor, wav):
    print("Generation started.", flush=True)
    reply, _ = complete(model, processor, wav, speak=False)
    print(f"model: {reply}", flush=True)
    calls = tool_calls(reply)
    if calls:
        results = [{"name": name, "result": run_tool(name, args)} for name, args in calls]
        follow = "<tool_response>\n" + json.dumps(results) + "\n</tool_response>\nSpeak one short sentence confirming the reading lamp. If a result has an error, say it could not be reached."
    else:
        print("no tool call", flush=True)
        follow = "You did not call a tool. Say you did not catch a request for the reading lamp. Do not claim the lamp changed."
    text, samples = complete(model, processor, wav, True, reply, follow)
    if samples is None or len(samples) == 0:
        print("model returned no audio", flush=True)
        return
    # The reply is not chunked: generate() returns the whole waveform, then playback starts.
    play_or_save(samples)
    if text:
        print(text, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Spoken request for the living-room reading lamp.")
    parser.add_argument("wav", nargs="?", help="Wav path. Omit to keep listening on the microphone.")
    given = parser.parse_args().wav
    if given:
        wav = Path(given).expanduser().resolve()
        if not wav.is_file():
            sys.exit(f"no such wav: {wav}")
    elif device_index("in") is None:
        sys.exit("no wav path, and no microphone is present")
    else:
        wav = None
    print(f"loading {MODEL_ID} on cpu", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa").eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    if wav is not None:
        one_turn(model, processor, wav)
        return
    try:
        while True:
            captured = record_until_silence()
            try:
                one_turn(model, processor, captured)
            finally:
                captured.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\nstopped", flush=True)

if __name__ == "__main__":
    main()
