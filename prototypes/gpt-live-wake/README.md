# Moved

The app is `home/`. `python lights.py` here still starts it. Edit `home/home.json`.

# Wake word, then a realtime lamp session

Sits on the microphone for free. When it hears the wake word anywhere in the sentence, it opens one `gpt-realtime-2.1-mini` session, sends the speech it already buffered (so "turn off the lights, Stanley" still includes the request), then keeps the mic live.

The session ends when someone says goodbye, thank you, okay good, shut up, stop talking, be quiet, or go away, even while it is still speaking. It also ends 20 seconds after the last thing it actually did: a lamp change, a spoken answer, or someone still talking. Audio from Stanley, and a reply still being produced, keep that clock reset. A rising bloop means engaged. A falling bloop means back to listening.

The wake spotter is local Vosk. Nothing is sent to OpenAI until the name is heard. The first run downloads the small English model into `.wake-model/`.

```bash
cd prototypes/gpt-live-wake
python -m pip install -r requirements.txt
cp .env.example .env
python lights.py
```

`home.json` is the configuration. `agent.name` is who it is, `agent.wake` is the word that opens a session, `agent.voice` is the OpenAI voice, and `agent.buffer_seconds` is how much talk to keep before that word. `lights` is each light's name, IP address, and where it is. That list is what it is told and what `set_light` accepts. Ctrl-C quits.
