# Home

Stanley listens locally. Saying the wake word opens one realtime session. Lights and doors are tools, so the session code does not know how a device works.

From the repo root, with the `stanley` environment:

```bash
python -m pip install -r home/requirements.txt
python -m home
```

From inside `home/`:

```bash
python start.py
```

The OpenAI key already in `prototypes/gpt-live-wake/.env` is used if `home/.env` has none. `home.json` is the house: the agent's name, wake word, voice, how many seconds of talk to keep, the lights, and the doors.

`set_door` talks to MyQ with `MYQ_USERNAME` and `MYQ_PASSWORD`. Chamberlain blocks unofficial clients, and Tesla is an official partner, so that login can fail even when the Tesla app works. A later local opener replaces `home/tools/garage.py` only.

A shower, when it is added, is another file in `home/tools`.
