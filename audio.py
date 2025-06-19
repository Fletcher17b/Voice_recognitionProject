import numpy as np
import os
import sounddevice as sd
samplerate = 16000 

current_dir = os.path.abspath(__file__)
current_dir = current_dir+"/AudioSamples"
os.chdir(current_dir)


def measure_ambient_noise(seconds=2):
    print(f"🎙️ Measuring ambient noise for {seconds} seconds...")
    duration_samples = int(seconds * samplerate)
    recording = sd.rec(duration_samples, samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    float_audio = recording.flatten().astype(np.float32) / 32768.0
    avg_volume = np.mean(np.abs(float_audio))
    print(f"🔈 Ambient noise level: {avg_volume:.5f}")
    return avg_volume