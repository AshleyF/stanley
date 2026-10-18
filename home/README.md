# Home

Stanley listens locally. Saying the wake word opens one realtime session. Lights and the shower are tools, so the session code does not know how a device works.

From the repo root, with the `stanley` environment:

```bash
python -m pip install -r home/requirements.txt
python -m home
```

From inside `home/`:

```bash
python start.py
```

The OpenAI key already in `prototypes/gpt-live-wake/.env` is used if `home/.env` has none. `home.json` is the house: the agent's name, wake word, voice, how many seconds of talk to keep, the lights, and the shower. Preset names and `max_temperature` live on the shower entry. Temperatures are Fahrenheit, and a request above that maximum is lowered to it.

`set_shower` uses `MOEN_EMAIL` and `MOEN_PASSWORD`. It is the same Moen cloud and Pusher commands as `moen/moen_api.py` in [AshleyF/sandbox](https://github.com/AshleyF/sandbox/tree/main/moen). Each command logs in and disconnects, so idle listening holds no shower connection.

`home/tools/garage.py` is still there and is not loaded. On 18 October 2026 MyQ returned 401 from `https://partner-identity.myq-cloud.com/connect/token`. The official app and Tesla still work. A local opener replaces that file, and `load_tools` is what offers it again. The door entry in `home.json` stays.
