from faster_whisper import WhisperModel
from ArduinoCommands import handle_command
from scipy.spatial.distance import cosine
from resemblyzer import VoiceEncoder, preprocess_wav
from sentence_transformers import SentenceTransformer, util
from utility import is_similar_command

import datetime
import sounddevice as sd
import queue
import numpy as np
import sys


model = WhisperModel("base", compute_type="float16")  
encoder = VoiceEncoder()

q = queue.Queue()
samplerate = 16000  
block_duration = 0.5  
device_index = 1

key_word = "artemis"
keyword_override = False

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_chunk = np.squeeze(indata.copy())
    q.put(audio_chunk)

stream = sd.InputStream(device=device_index, samplerate=samplerate, channels=1, dtype='int16', blocksize=int(block_duration * samplerate), callback=audio_callback)

with stream:
    buffer = np.empty((0,), dtype=np.float32)

    print("🎧 Listening...")
    while True:
        chunk = q.get()
        float_chunk = chunk.astype(np.float32) / 32768.0 
        buffer = np.concatenate((buffer, float_chunk))

        if len(buffer) >= 2 * samplerate:
            segment_audio = buffer[:2 * samplerate]
            buffer = buffer[int(0.5 * samplerate):]  

            segments, _ = model.transcribe(segment_audio, language='es', beam_size=5,vad_filter=True, vad_parameters={"threshold": 0.5})

            for segment in segments:
                text = segment.text.strip().lower()
                print(f"🗣️ Transcript: {text}")
                handle_command(text)

                
