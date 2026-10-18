# Home

Stanley listens locally. Saying the wake word opens one realtime session. Lights and doors are tools, so the session code does not know how a device works.

```bash
cd home
python -m pip install -r requirements.txt
cp .env.example .env
python -m home
```

The OpenAI key already in `prototypes/gpt-live-wake/.env` is used if `home/.env` has none. `home.json` is the house: the agent's name, wake word, voice, how many seconds of talk to keep, the lights, and the doors.

`set_door` talks to MyQ with `MYQ_USERNAME` and `MYQ_PASSWORD`. Chamberlain blocks unofficial clients, and Tesla is an official partner, so that login can fail even when the Tesla app works. A later local opener replaces `home/tools/garage.py` only.

A shower, when it is added, is another file in `home/tools`.
