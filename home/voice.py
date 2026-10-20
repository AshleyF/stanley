"""Who is speaking. A pretrained speaker model, plus one embedding per person.

Enrollment is the training: a few clips of each person. Each clip becomes a
vector. The saved vector is their average. A new utterance is the same kind of
vector. The closest enrolled voice wins unless Someone is more likely.
Someone is the unlikely class: most speech is one of the enrolled voices.
"""

import asyncio
import sys
import wave
from pathlib import Path

import numpy as np

from .config import DATA
from .log import record

RATE = 16000
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TEMPERATURE = 0.16


def note(text):
    print(f"\n[{text}]", flush=True)
    record("voice", text)


def unit(vector):
    values = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(values))
    if length == 0:
        return values
    return values / length


def entry_line(threshold, prior, count):
    """Cosine where an enrolled voice beats Someone, given the household prior."""
    household = min(0.99, max(0.0, float(prior)))
    each = household / max(count, 1)
    someone = 1.0 - household
    if each <= 0 or someone <= 0 or count <= 0:
        return threshold
    return threshold + TEMPERATURE * float(np.log(someone / each))


def rank_voices(gallery, vector):
    query = unit(vector)
    if float(np.linalg.norm(query)) == 0:
        return []
    return sorted(((float(query @ sample), name) for name, sample in gallery.items()), reverse=True)


def best_match(gallery, vector, threshold, prior=0.95):
    """gallery is {name: unit vector}. Returns (name or None, score)."""
    scored = rank_voices(gallery, vector)
    if not scored:
        return None, None
    score, name = scored[0]
    if score < entry_line(threshold, prior, len(scored)):
        return None, score
    return name, score


def decide(scored, threshold, prior, current):
    """Who is speaking, with a sticky name.

    From Someone, the closest voice wins once it crosses the prior line.
    After that, a short or weak clip keeps the same person. Leaving them
    takes a different enrolled voice at or above `threshold`, a positive match.
    Returns (name, score, held).
    """
    if not scored:
        if current and current != "Someone":
            return current, None, True
        return "Someone", None, False
    score, name = scored[0]
    if current and current != "Someone":
        if name != current and score >= threshold:
            return name, score, False
        own = next((s for s, n in scored if n == current), score)
        raw = name if score >= entry_line(threshold, prior, len(scored)) else "Someone"
        return current, own, raw != current
    if score < entry_line(threshold, prior, len(scored)):
        return "Someone", score, False
    return name, score, False


class Voices:
    def __init__(self, directory, names, match, prior=0.95):
        self.directory = Path(directory)
        self.names = list(names)
        self.match = float(match)
        self.prior = float(prior)
        self.gallery = {}
        self._model = None

    def load(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gallery = {}
        for name in self.names:
            path = self.directory / f"{name}.vec"
            if not path.is_file():
                continue
            vector = np.fromfile(path, dtype=np.float32)
            if vector.size:
                self.gallery[name] = unit(vector)
        if self.gallery:
            note("voice: " + ", ".join(self.gallery))
        else:
            note("voice: no voices enrolled")

    async def identify(self, pcm, rate=24000, current=None):
        if not self.gallery or len(pcm) < rate:
            return decide([], self.match, self.prior, current)
        try:
            vector = await asyncio.to_thread(self.embed_pcm, pcm, rate)
            scored = rank_voices(self.gallery, vector)
            name, score, held = decide(scored, self.match, self.prior, current)
        except Exception as exc:
            record("voice", f"identify failed: {exc}")
            name, score, held = decide([], self.match, self.prior, current)
        return name, score, held

    def embed_pcm(self, pcm, rate):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != RATE and len(samples):
            count = int(len(samples) * RATE / rate)
            samples = np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples).astype(np.float32)
        return self._embed(samples)

    def _embed(self, samples):
        import torch
        model = self._load_model()
        audio = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            vector = model.encode_batch(audio)
        return unit(vector.squeeze().cpu().numpy())

    def _load_model(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            saved = self.directory / "model"
            saved.mkdir(parents=True, exist_ok=True)
            self._model = EncoderClassifier.from_hparams(source=MODEL, savedir=str(saved))
        return self._model

    def enroll_wav(self, name, path):
        if name not in self.names:
            raise SystemExit(f"{name} is not in people in home.json")
        samples = read_wav(Path(path))
        folder = self.directory / name
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / Path(path).name
        if dest.resolve() != Path(path).resolve():
            dest.write_bytes(Path(path).read_bytes())
        self._rebuild(name)
        return dest

    def enroll_live(self, name, clips=3, seconds=20):
        if name not in self.names:
            raise SystemExit(f"{name} is not in people in home.json")
        import sounddevice as sd
        folder = self.directory / name
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*.wav"))
        for index in range(clips):
            print(f"\nClip {index + 1} of {clips}. Talk as {name} for {seconds} seconds.", flush=True)
            input("Press Enter, then talk. ")
            audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
            sd.wait()
            path = folder / f"{len(existing) + index + 1:02d}.wav"
            write_wav(path, audio.reshape(-1).tobytes())
            print(f"saved {path.name}", flush=True)
        self._rebuild(name)

    def _rebuild(self, name):
        vectors = []
        for path in sorted((self.directory / name).glob("*.wav")):
            vectors.append(self._embed(read_wav(path)))
        if not vectors:
            raise SystemExit(f"no clips for {name}")
        mean = unit(np.mean(np.stack(vectors), axis=0))
        (self.directory / f"{name}.vec").write_bytes(mean.astype(np.float32).tobytes())
        self.gallery[name] = mean
        print(f"enrolled {name} from {len(vectors)} clips", flush=True)


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit(f"{path} must be 16-bit wav")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(samples):
        count = int(len(samples) * RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples)
    return samples.astype(np.float32)


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm)


def main(argv):
    from .config import load_home
    if len(argv) < 2 or argv[0] != "enroll":
        raise SystemExit("usage: python -m home.voice enroll Name [clip.wav ...]")
    name, wavs = argv[1], argv[2:]
    home = load_home()
    voices = Voices(DATA / "voices", home["people"], home["voice_match"], home["voice_prior"])
    voices.load()
    if wavs:
        for path in wavs:
            voices.enroll_wav(name, path)
    else:
        voices.enroll_live(name)


if __name__ == "__main__":
    main(sys.argv[1:])
