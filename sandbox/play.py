import pyaudio
import wave

# Open the recorded file
wf = wave.open("test_recording.wav", 'rb')

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for playback
stream = p.open(format=pyaudio.paInt16,
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# Read data from the file and play it back
data = wf.readframes(1024)

while data:
    stream.write(data)
    data = wf.readframes(1024)

# Close the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
