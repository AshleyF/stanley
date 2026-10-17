# Qwen Omni lights

Local speech in, speech out, and lights that live only in memory. The model is `Qwen/Qwen2.5-Omni-3B`, the smallest official Qwen Omni checkpoint that hears speech and speaks back. The newer local Qwen3-Omni checkpoint is `Qwen/Qwen3-Omni-30B-A3B-Instruct`, which is a larger mixture-of-experts model, so this prototype stays on the 3B.

On a Mac, create a new Python 3.12 environment. Install torch and torchvision from PyPI so the build includes MPS, then the rest of the pins.

```bash
conda create -n qwen-lights python=3.12
conda activate qwen-lights
pip install torch==2.14.1 torchvision==0.29.1
pip install -r requirements.txt
python lights.py request.wav
```

Leave off the wav path to record from the microphone when a capture device is present. The reply plays on an output device when one exists. Otherwise it is written to `reply.wav` in the working directory. Microphone and speaker playback need PortAudio (`brew install portaudio` on a Mac, `libportaudio2` on Linux).

On a Linux machine with no GPU, install the CPU wheels from the PyTorch CPU index instead of the PyPI torch line, then the same requirements file:

```bash
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Tools are `lights_on`, `lights_off`, and `set_light_color`. Each call prints one line, such as `lights on, warm white`, `lights on, purple`, or `lights off`, and returns the in-memory state. Changing the color turns the lights on. Turning them off keeps the last color and prints `lights off`.

The weights are about 12 GB. Qwen's published BF16 floor for this checkpoint is about 18 GB. Attention uses PyTorch SDPA, not flash-attn.
