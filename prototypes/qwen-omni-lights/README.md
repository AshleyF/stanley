# Qwen Omni lights

Speech in, speech out, and the living-room reading bulb. The model is `Qwen/Qwen2.5-Omni-3B`, the smallest official Qwen Omni checkpoint that hears speech and speaks back. The newer local Qwen3-Omni checkpoint is `Qwen/Qwen3-Omni-30B-A3B-Instruct`, which is a larger mixture-of-experts model, so this prototype stays on the 3B.

The bulb is a Kasa SmartBulb at `192.168.4.38` on the house LAN. On, off, and color. A color turns it on and sets HSV. Names are purple, red, blue, green, and warm white. `python-kasa` is the client.

On a Mac, create a new Python 3.12 environment. Install torch and torchvision from PyPI so the build includes MPS, then the rest of the pins, including python-kasa.

```bash
conda create -n qwen-lights python=3.12
conda activate qwen-lights
pip install torch==2.14.1 torchvision==0.29.1
pip install -r requirements.txt
python lights.py request.wav
```

Leave off the wav path to record from the microphone when a capture device is present. The model loads once, then it keeps listening: a short silence ends that request, the reply plays, and it listens again until Ctrl-C. A wav path is a single turn. It is not full duplex. This transformers build returns the spoken reply as one waveform at the end of `generate()`, and playback starts then. Otherwise the reply is written to `reply.wav` in the working directory. Microphone and speaker playback need PortAudio (`brew install portaudio` on a Mac, `libportaudio2` on Linux).

On a Linux machine with no GPU, install the CPU wheels from the PyTorch CPU index instead of the PyPI torch line, then the same requirements file:

```bash
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`lights_on`, `lights_off`, and `set_light_color` are registered as tool schemas on the text pass. The microphone turn is only the audio. This checkpoint's chat template has no `tools` argument, so the schema is appended to the system message in Qwen's tool format, and the script intercepts `<tool_call>`. A color prints `lights on, purple` (or red, blue, green, warm white). Off prints `lights off`. If the bulb cannot be reached, the error is printed and the spoken reply still runs.

The weights are about 12 GB. Qwen's published BF16 floor for this checkpoint is about 18 GB. Attention uses PyTorch SDPA, not flash-attn.
