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

Copy `home/.env.example` to `home/.env` and set `XAI_API_KEY`. That file stays on this machine. Stanley talks through Grok speech-to-speech (`grok-voice-3.0`). `home.json` is the house: the agent's name, wake word, voice, how many seconds of talk to keep, the lights, and the shower. `agent.voice` is a Grok voice, such as `rex`. Preset names and `max_temperature` live on the shower entry. Temperatures are Fahrenheit, and a request above that maximum is lowered to it.

`set_shower` uses `MOEN_EMAIL` and `MOEN_PASSWORD`. It is the same Moen cloud and Pusher commands as `moen/moen_api.py` in [AshleyF/sandbox](https://github.com/AshleyF/sandbox/tree/main/moen). Each command logs in and disconnects, so idle listening holds no shower connection.

A timer is local. "Remind me in 20 minutes to check the hot tub" or "set a timer for 10 minutes" calls `set_timer`, and the clock keeps running with no session open. When it is set, the model writes the sentence to speak later, with the wait kept separate from the task. When it is due, Home opens one session and Stanley says that sentence. Asking what reminders are set calls `list_reminders`, including ones from earlier. "Okay" or "thanks" clears it. If nobody answers, it speaks again later: about half a minute for a short timer, a couple of minutes for a longer one. The waiting itself stays on the clock in `home/timers.json`.

`agent.transcript_minutes` is how much recent talk is read into the next session. It starts at 10. Stanley's lines are his own transcript. What you said is transcribed on that same session, only while it is open. Set it to 0 to keep none of it out of the next session. The file still keeps about an hour, for issue reports.

`home/logs/` has one file per session, named by when that session started, such as `20261019-010530.log`. Each line has its own time. The file is not read back into the session.

Code is ephemeral. `run_script` is how Stanley does something the other tools do not. Home saves it in `home/ephemeral/` and runs it without blocking the session. Lines it prints show up while it is still running, and the final output or error comes back when it finishes. `stop_script` stops one. Those files stay until you delete them. He does not save lasting tools. The prompt includes how to call the lights and the shower from that code. A reminder still only speaks.

A deeper feature of Home, or something wrong with Stanley, is a file in `home/issues/`. He asks until he understands, then writes that description, the last hour of transcript, and the last hour of logs. Those files stay on this machine. He also writes one when you tell him to file an issue.

`home/tools/garage.py` is still there and is not loaded. On 18 October 2026 MyQ returned 401 from `https://partner-identity.myq-cloud.com/connect/token`. The official app and Tesla still work. A local opener replaces that file, and `load_tools` is what offers it again. The door entry in `home.json` stays.
