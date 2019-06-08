"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from pprint import pprint
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

pprint(p.get_default_input_device_info())
pprint(p.get_default_host_api_info())

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    while stream.get_read_available() == 0:
        print("no input available")
        time.sleep(CHUNK / RATE)
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

(r, d) = wavfile.read('output.wav')
(spectrum, freqs, t, im) = plt.specgram(d[:,0], NFFT=1024, Fs=r, scale_by_freq=False)
plt.show()



