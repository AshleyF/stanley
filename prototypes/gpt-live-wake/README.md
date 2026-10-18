# Wake word, then a realtime lamp session

Sits on the microphone for free. When it hears **Stanley** anywhere in the sentence, it opens one `gpt-realtime-2.1` session, sends the speech it already buffered (so "turn off the lights, Stanley" still includes the request), then keeps the mic live.

The session ends when someone says goodbye, shut up, stop talking, be quiet, or go away, even while Stanley is still speaking. It also ends 12 seconds after the last thing Stanley actually did: a lamp change or a spoken answer. Background audio does not keep it open. A rising bloop means engaged. A falling bloop means back to listening.

The wake spotter is local Vosk. Nothing is sent to OpenAI until the name is heard. The first run downloads the small English model into `.wake-model/`.

```bash
cd prototypes/gpt-live-wake
python -m pip install -r requirements.txt
cp .env.example .env
python lights.py
```

The lights are `lights.json`: a name, an IP address, and a description of where it is. That list is what Stanley is told and what `set_light` accepts. Ctrl-C quits.
