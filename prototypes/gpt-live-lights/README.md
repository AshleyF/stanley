# GPT Live lights

One realtime session for the living-room reading lamp at `192.168.4.38`. The model is `gpt-realtime-2.1`. It hears, speaks, and calls `set_lamp` itself. This script runs that call with `python-kasa`. The session stays open.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The `.env` file stays on this machine.

```bash
conda activate stanley
pip install -r requirements.txt
python lights.py
```

Headphones keep the speaker out of the microphone. Ctrl-C ends the session. PortAudio is required (`brew install portaudio` on a Mac).
