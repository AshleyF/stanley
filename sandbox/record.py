import pyaudio
import wave

# Set up audio parameters (adjust based on your microphone's capabilities)
FORMAT = pyaudio.paInt16  # Sample format (16-bit)
CHANNELS = 1              # Number of audio channels (1 for mono)
RATE = 44100              # Sample rate (44.1kHz is standard)
CHUNK = 1024              # Size of each audio chunk
SECONDS = 5               # Duration of the recording
OUTPUT_FILENAME = "test_recording.wav"  # Output file name

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream for the microphone
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record audio
for _ in range(0, int(RATE / CHUNK * SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# Stop the stream and close it
stream.stop_stream()
stream.close()

# Write the recorded data to a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Close PyAudio
p.terminate()

print(f"Recording saved as {OUTPUT_FILENAME}")
