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

Copy `home/.env.example` to `home/.env` and set `XAI_API_KEY`. That file stays on this machine. Stanley talks through Grok speech-to-speech (`grok-voice-3.0`). Runtime files live in `home/data/`: the transcript, the timers, the logs, the memories, the issues, and the ephemeral scripts. `home.json` is the house: the agent's name, wake word, voice, how many seconds of talk to keep, the lights, and the shower. `agent.pause_seconds` is a quiet gap. Talk from before the last gap of that length is not sent with the wake word, so an earlier conversation stays out. It starts at 8. A shorter pause, as in "turn off the lights, Stanley", stays. `agent.voice` is a Grok voice, such as `rex`. Preset names and `max_temperature` live on the shower entry. Temperatures are Fahrenheit, and a request above that maximum is lowered to it.

`set_shower` uses `MOEN_EMAIL` and `MOEN_PASSWORD`. It is the same Moen cloud and Pusher commands as `moen/moen_api.py` in [AshleyF/sandbox](https://github.com/AshleyF/sandbox/tree/main/moen). Each command logs in and disconnects, so idle listening holds no shower connection.

A timer is local. "Remind me in 20 minutes to check the hot tub" or "set a timer for 10 minutes" calls `set_timer`, and the clock keeps running with no session open. When it is set, the model writes the sentence to speak later, with the wait kept separate from the task. When it is due, Home opens one session and Stanley says that sentence. Asking what reminders are set calls `list_reminders`, including ones from earlier. "Okay" or "thanks" clears it. If nobody answers, it speaks again later: about half a minute for a short timer, a couple of minutes for a longer one. The waiting itself stays on the clock in `home/data/timers.json`.

`agent.transcript_minutes` is how much recent talk is pasted into the next session, verbatim. It starts at 1. That is so a session closed and opened again a moment later is still the same conversation. Set it higher to carry more. Set it to 0 to paste none. Older than that window, a note comes back only when what was just said is similar enough, and that note is the digested memory, not the transcript. The file still keeps about an hour, for issue reports. Stanley's lines are his own transcript. What you said is transcribed on that same session, only while it is open.

`home/data/logs/` has one file per session, named by when that session started, such as `20261019-010530.log`. Each line has its own time. The file is not read back into the session.

Code is ephemeral. `run_script` is how Stanley does something the other tools do not. Home saves it in `home/data/ephemeral/` and runs it without blocking the session. Lines it prints show up while it is still running. Ordinary output is for him to remember, and he does not talk over it. He speaks if that output is an error or something extraordinary, or if you ask what is going on. The final output or error comes back when it finishes. `stop_script` stops one. Those files stay until you delete them. He does not save lasting tools. The prompt includes how to call the lights and the shower from that code. A reminder still only speaks.

Memory is a directory of notes, `home/data/memories/`. Each note is a `.txt` file and a `.mem` file beside it. The text is what he remembers. The `.mem` file is the embedding, and the length of that vector. Before two of them are compared, the direction they all share is removed. This model points every sentence the same way, so a raw cosine calls a greeting and a birthday similar. What is left is the difference between them. The model is `memory.embed_model`. A name with a slash, such as `intfloat/e5-small-v2`, runs locally. The first start downloads it. This key's xAI account has no embedding model, so that local model is the one in `home.json`. A name without a slash is sent to the xAI embeddings API instead. Home reads the directory when it starts and builds a small-world search index in memory. A text file with no embedding, or text that has been edited, is embedded again. An embedding whose text file is gone is deleted. Add a text file, edit one, or delete a pair. The next start picks that up.

At the end of a session the transcript, the things he did, and the code he wrote go to `grok-4.7`. That model splits the session into one memory per topic. It keeps a quotation or a piece of code verbatim when that is the part worth keeping, especially code you said worked. Each memory starts with `From:` and the time of the conversation.

Before a new memory is saved, Home looks for notes already close to it. Close ones are merged by the same model into one note that keeps what is new and drops the overlap. Every `From:` line stays, so a memory can name two conversations. `memory.merge` is how close they have to be, on that same comparison. It starts at 0.62.

During a session, a finished thing you said and a finished line from Stanley are embedded as queries. Notes closer than `memory.recall` (0.50 after that shared direction is removed) are placed into the conversation. A note a little below that, down to 0.06 under it, is shown to `grok-4.7` first. It stays out unless that model says it answers what was just said, so a greeting does not carry an unrelated note, and a question can still find a fact the cutoff would have dropped. He may bring one up when it is clearly worth saying. Most of the time he says nothing and keeps it in mind, including for when you ask. The `From:` line is how he can tell you it was last Wednesday.

`people` in `home.json` is who lives here. It starts as Ashley and Tracey. `voice_match` is the cosine where an enrolled voice and Someone would tie if they were equally likely. `voice_prior` is the chance the speaker is one of the enrolled voices, split across them. It starts at 0.95, so Someone is the unlikely class and a close-enough match is named. The recognizer is a pretrained speaker model. Enrollment is the part that learns these voices: `python -m home.voice enroll Ashley`, then the same for Tracey, while Home is not running. Each person talks for three clips of about twenty seconds. The clips and the averaged voice vector stay in `home/data/voices/`. A finished utterance is matched against those vectors, the console names them, and the transcript uses that name. Stanley waits for that match before he answers, and he answers once. The reply instructions tell him who is speaking, and that is the name he uses. Naming someone is the low bar. After that, a short or weak follow-up keeps them. The name changes only when a different enrolled voice clears `voice_match`, a positive match. It does not fall back to Someone in between.

Each person also has a preference note, `home/data/people/Ashley.txt` and the same for Tracey. It is a short description that is allowed to change, not a memory of an event. After a session a separate call to `grok-4.7` sees the session and the current notes and writes each note again: a new preference is added, a corrected one replaces the old, and the rest stays. Only what that person said, or claimed as their own, changes their note. When they ask for the shower and do not give a temperature or a preset, Stanley uses the note of whoever is speaking.

A deeper feature of Home, or something wrong with Stanley, is a file in `home/data/issues/`. He asks until he understands, then writes that description, the last hour of transcript, and the last hour of logs. Those files stay on this machine. He also writes one when you tell him to file an issue.

`home/tools/garage.py` is still there and is not loaded. On 18 October 2026 MyQ returned 401 from `https://partner-identity.myq-cloud.com/connect/token`. The official app and Tesla still work. A local opener replaces that file, and `load_tools` is what offers it again. The door entry in `home.json` stays.
