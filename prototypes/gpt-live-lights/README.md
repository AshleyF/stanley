# GPT Live lights

One realtime session for the living-room reading lamp at `192.168.4.38`. The model is `gpt-realtime-2.1`. `set_lamp` takes on/off and HSV. The model chooses the numbers. Speech that is not about the lamp calls `wait_for_user` and stays quiet. This script runs the lamp call with `python-kasa`. The session stays open.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The `.env` file stays on this machine.

```bash
conda activate stanley
pip install -r requirements.txt
python lights.py
```

Headphones keep the speaker out of the microphone. Ctrl-C ends the session. PortAudio is required (`brew install portaudio` on a Mac).
