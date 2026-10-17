"""Speech in, speech out, and the living-room reading bulb.

The checkpoint is Qwen/Qwen2.5-Omni-3B. The bulb is the Kasa color lamp at
192.168.4.204 (reading / overhanging / black).
"""

import argparse
import asyncio
import contextlib
import json
import sys
import tempfile
import wave
from pathlib import Path

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
BULB_HOST = "192.168.4.204"
SAMPLE_RATE_OUT = 24000
# Required by the talker. Extra instructions live in the user turn so this
# sentence stays intact and speech generation still runs.
SYSTEM_TEXT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# hue, saturation, value. Saturation and value are percentages.
COLORS = {
    "purple": (280, 100, 100),
    "red": (0, 100, 100),
    "blue": (240, 100, 100),
    "green": (120, 100, 100),
    "warm white": (30, 20, 100),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lights_on",
            "description": "Turn the living-room reading lamp on. Optional color: purple, red, blue, green, or warm white.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "purple, red, blue, green, or warm white. Setting a color also turns the lamp on.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lights_off",
            "description": "Turn the living-room reading lamp off.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_light_color",
            "description": "Set the living-room reading lamp color and turn it on. Colors: purple, red, blue, green, warm white.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "purple, red, blue, green, or warm white.",
                    }
                },
                "required": ["color"],
            },
        },
    },
]


def color_name(value):
    text = " ".join(str(value).split()).lower()
    if text not in COLORS:
        raise ValueError(f"unknown color: {text or value}")
    return text


def _error_text(exc):
    return str(exc).strip() or type(exc).__name__


async def drive(action, color=None):
    """On, off, or color for the one reading bulb. A network error is printed and returned."""
    from kasa import DeviceConfig, Module, SmartBulb

    bulb = None
    try:
        bulb = SmartBulb(BULB_HOST, config=DeviceConfig(host=BULB_HOST, timeout=5))
        await bulb.update()
        if action == "off":
            await bulb.turn_off()
            print("lights off", flush=True)
            return {"on": False}
        await bulb.turn_on()
        if color:
            name = color_name(color)
            hue, saturation, value = COLORS[name]
            await bulb.modules[Module.Light].set_hsv(hue, saturation, value)
            print(f"lights on, {name}", flush=True)
            return {"on": True, "color": name, "hsv": [hue, saturation, value]}
        print("lights on", flush=True)
        return {"on": True}
    except Exception as exc:
        text = _error_text(exc)
        print(text, flush=True)
        return {"error": text}
    finally:
        if bulb is not None:
            with contextlib.suppress(Exception):
                await bulb.disconnect()


def run_tool(name, args):
    try:
        if name == "lights_off":
            return asyncio.run(drive("off"))
        if name == "lights_on":
            color = args.get("color")
            if color:
                color_name(color)
            return asyncio.run(drive("on", color))
        if name == "set_light_color":
            if not args.get("color"):
                print("color is required", flush=True)
                return {"error": "color is required"}
            color_name(args["color"])
            return asyncio.run(drive("on", args["color"]))
    except ValueError as exc:
        text = _error_text(exc)
        print(text, flush=True)
        return {"error": text}
    except Exception as exc:
        text = _error_text(exc)
        print(text, flush=True)
        return {"error": text}
    return {"error": f"unknown tool {name}"}


def tool_instructions():
    signatures = "\n".join(json.dumps(tool) for tool in TOOLS)
    return (
        "The attached audio is a person talking to the house.\n"
        "The only lamp is the living-room reading lamp.\n"
        "You may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{signatures}\n</tools>\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
        "Call only those tools. Do not describe the call in any other format."
    )


def parse_tool_calls(text):
    """Pull tool calls out of the reply. The model sometimes repeats the opening tag."""
    decoder = json.JSONDecoder()
    calls = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = end
        if not isinstance(obj, dict) or not obj.get("name"):
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append((obj["name"], args))
    return calls


def input_devices():
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    found = []
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            found.append((index, device))
    return found


def output_device_index():
    try:
        import sounddevice as sd
    except Exception:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for index, device in enumerate(devices):
        if device.get("max_output_channels", 0) > 0:
            return index
    return None


def record_mic(path, seconds=5, rate=16000):
    import sounddevice as sd

    frames = int(seconds * rate)
    print(f"Recording {seconds} seconds...", flush=True)
    audio = sd.rec(frames, samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(audio.tobytes())
    print(f"Recorded {path}", flush=True)


def play_or_save(samples, out_path):
    import numpy as np
    import soundfile as sf

    samples = np.asarray(samples, dtype="float32").reshape(-1)
    device = output_device_index()
    if device is None:
        sf.write(out_path, samples, SAMPLE_RATE_OUT)
        print(f"wrote {out_path}", flush=True)
        return
    import sounddevice as sd

    sd.play(samples, SAMPLE_RATE_OUT, device=device)
    sd.wait()
    print("played reply", flush=True)


def prepare(processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point():
            inputs[key] = value.to(device="cpu", dtype=torch.bfloat16)
        else:
            inputs[key] = value.to(device="cpu")
    return inputs


def new_token_ids(generated, prompt_len):
    text_ids = generated[0] if isinstance(generated, tuple) else generated
    if hasattr(text_ids, "sequences"):
        text_ids = text_ids.sequences
    if text_ids.ndim == 1:
        text_ids = text_ids.unsqueeze(0)
    # generate() returns the prompt plus new tokens. A text-only return that is
    # already shorter than the prompt is just the new tokens.
    if text_ids.shape[-1] > prompt_len:
        text_ids = text_ids[:, prompt_len:]
    return text_ids


def decode(processor, generated, prompt_len):
    text = processor.batch_decode(
        new_token_ids(generated, prompt_len),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text[0] if text else ""


def audio_from(generated):
    if not isinstance(generated, tuple) or len(generated) < 2:
        return None
    audio = generated[1]
    if audio is None:
        return None
    if isinstance(audio, (list, tuple)):
        audio = audio[0] if audio else None
    if audio is None:
        return None
    return audio.reshape(-1).detach().to(torch.float32).cpu().numpy()


def generate(model, inputs, *, speak):
    kwargs = {
        "thinker_do_sample": False,
        "thinker_max_new_tokens": 64 if speak else 256,
        "generation_mode": "audio" if speak else "text",
    }
    if speak:
        kwargs["speaker"] = "Chelsie"
        kwargs["talker_do_sample"] = False
        kwargs["talker_max_new_tokens"] = 512
    with torch.inference_mode():
        return model.generate(**inputs, **kwargs)


def load_wav(path):
    """16 kHz mono float32. soundfile reads the file; numpy resamples if needed."""
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sample_rate != 16000 and len(data):
        new_len = max(1, int(round(len(data) * 16000 / sample_rate)))
        old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        data = np.interp(new, old, data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def messages_for(wav_path, extra_user=None, earlier=None):
    user = [{"type": "text", "text": tool_instructions()}, {"type": "audio", "audio": load_wav(wav_path)}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": user},
    ]
    if earlier:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": earlier}]})
    if extra_user:
        messages.append({"role": "user", "content": [{"type": "text", "text": extra_user}]})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Omni-3B on a spoken request for the reading lamp.")
    parser.add_argument(
        "wav",
        nargs="?",
        help="Spoken request as a wav file. Omit to use the microphone when one is present.",
    )
    args = parser.parse_args()

    if args.wav:
        wav_path = Path(args.wav).expanduser().resolve()
        if not wav_path.is_file():
            sys.exit(f"no such wav: {wav_path}")
        cleanup = None
    else:
        if not input_devices():
            sys.exit("no wav path, and no microphone is present")
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()
        wav_path = Path(temp.name)
        record_mic(wav_path)
        cleanup = wav_path

    try:
        print(f"loading {MODEL_ID} on cpu", flush=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="sdpa",
        )
        model.eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

        first_inputs = prepare(processor, messages_for(wav_path))
        first = generate(model, first_inputs, speak=False)
        reply = decode(processor, first, first_inputs["input_ids"].shape[-1]).strip()
        print(f"model: {reply}", flush=True)

        results = []
        for name, tool_args in parse_tool_calls(reply):
            results.append({"name": name, "result": run_tool(name, tool_args)})

        follow = (
            "Tool results:\n"
            + json.dumps(results)
            + "\nSpeak one short sentence confirming the reading lamp. "
            "If a result has an error, say it could not be reached. Do not call a tool."
        )
        spoken_inputs = prepare(processor, messages_for(wav_path, extra_user=follow, earlier=reply))
        spoken = generate(model, spoken_inputs, speak=True)
        spoken_text = decode(processor, spoken, spoken_inputs["input_ids"].shape[-1]).strip()
        if spoken_text:
            print(spoken_text, flush=True)
        samples = audio_from(spoken)
        if samples is None or len(samples) == 0:
            sys.exit("model returned no audio")
        play_or_save(samples, Path.cwd() / "reply.wav")
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
