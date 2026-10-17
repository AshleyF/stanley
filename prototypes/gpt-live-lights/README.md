# GPT Live lights

One GPT Live session for the living-room reading lamp at `192.168.4.38`. The session stays open. The voice model is `gpt-live-1`. Lamp changes are a `set_lamp` tool on the Responses backend, and this script runs that tool with `python-kasa`.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The `.env` file stays on this machine.

```bash
conda activate stanley
pip install -r requirements.txt
python lights.py
```

Headphones keep the speaker out of the microphone. Ctrl-C ends the session. PortAudio is required (`brew install portaudio` on a Mac).
