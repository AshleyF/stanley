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

A timer is local. "Remind me in 20 minutes to check the hot tub" or "set a timer for 10 minutes" calls `set_timer`, and the clock keeps running with no session open. When it is due, Home opens one session, tells the model when it was set and the sentence that set it, and Stanley says it out loud. Asking what reminders are set calls `list_reminders`, including ones from earlier. "Okay" or "thanks" clears it. If nobody answers, it speaks again later: about half a minute for a short timer, a couple of minutes for a longer one. The waiting itself stays on the clock in `home/timers.json`.

`agent.transcript_minutes` is how much recent talk is read into the next session. It starts at 10. Stanley's lines are his own transcript. What you said is transcribed on that same session, only while it is open. Set it to 0 to keep none of it.

`home/tools/garage.py` is still there and is not loaded. On 18 October 2026 MyQ returned 401 from `https://partner-identity.myq-cloud.com/connect/token`. The official app and Tesla still work. A local opener replaces that file, and `load_tools` is what offers it again. The door entry in `home.json` stays.
