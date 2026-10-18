# Wake word, then a realtime lamp session

Sits on the microphone for free. When it hears **Stanley** anywhere in the sentence, it opens one `gpt-realtime-2.1` session, sends the speech it already buffered (so "turn off the lights, Stanley" still includes the request), then keeps the mic live.

The session ends when the model calls `end_session` (goodbye, see you later, okay thanks), or 12 seconds after the last thing Stanley actually did: a lamp change or a spoken answer. Background audio does not keep it open. A rising bloop means engaged. A falling bloop means back to listening.

The wake spotter is local Vosk. Nothing is sent to OpenAI until the name is heard. The first run downloads the small English model into `.wake-model/`.

```bash
cd prototypes/gpt-live-wake
python -m pip install -r requirements.txt
cp .env.example .env
python lights.py
```

The reading lamp is the Kasa bulb at `192.168.4.38`. Ctrl-C quits.
