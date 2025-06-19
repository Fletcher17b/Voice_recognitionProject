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
transformer_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
encoder = VoiceEncoder()

q = queue.Queue()
samplerate = 16000  
block_duration = 0.5  
device_index = 1

key_word = "artemis"
keyword_override = False

intents = {
    "turn_on_lights": [
        "Está oscuro", "No veo nada", "Enciende la luz", "Pon la luz", "Ilumina el cuarto", "Prende la luz","Enciende la luz", "Esta algo oscuro aqui"
    ],
    "turn_off_lights": [
        "Apaga la luz", "Demasiada luz", "Quiero oscuridad", "Está muy brillante", "Apaga las luces"
    ]
    
}

""" "turn_on_fan": [
        "Hace calor", "Estoy sudando", "Enciende el ventilador", "Prende el abanico", "Esta haciendo calor"
    ],
    "turn_on_heater": [
        "Hace frío", "Estoy congelándome", "Prende la calefacción", "Enciende el calentador"
    ] """

""" 
         "turn_on_fan": "FAN_ON",
        "turn_on_heater": "HEATER_ON"
          """

intent_embeddings = []
intent_labels = []

for intent, phrases in intents.items():
    for phrase in phrases:
        intent_embeddings.append(transformer_model.encode(phrase))
        intent_labels.append(intent)

#wav = preprocess_wav("mario_sample.wav") 
#embedding = encoder.embed_utterance(wav)
#np.save("mario_embedding.npy", embedding)

def get_intent(user_phrase):
    embedding = transformer_model.encode(user_phrase)
    similarities = util.cos_sim(embedding, intent_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    return intent_labels[best_idx], similarities[best_idx].item()

def parse_command(intent,text):
    cmd_map = {
        "turn_on_lights": "LIGHT_ON",
        "turn_off_lights": "LIGHT_OFF",
        
    }

    command = cmd_map.get(intent)
    print("command:",command)
    if command:
        print(f"Sent: {command}")
        handle_command(command.strip())

""" known_speakers = {
    "Mario": np.load("mario_embedding.npy"),
    # "Ana": np.load("ana_embedding.npy")
}
 """
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_chunk = np.squeeze(indata.copy())
    q.put(audio_chunk)


def authorization_func(segment_audio,samplerate,text):
    live_wav = preprocess_wav(segment_audio, source_sr=samplerate)
    live_embedding = encoder.embed_utterance(live_wav)

    best_match = None
    best_similarity = 0

    #for name, emb in known_speakers.items():
    #    similarity = 1 - cosine(live_embedding, emb)
    #    if similarity > best_similarity:
    #        best_similarity = similarity
    #        best_match = name

        #if best_similarity > 0.75:
          #  print(f"👤 Speaker identified: {best_match} ({best_similarity:.2f})")

          #  AUTHORIZED_SPEAKER = best_match

          #  if is_authorized(best_match):
          #      handle_command(text)
          #  else:
          #      print("Unknown or unauthorized speaker")


def measure_ambient_noise(seconds=2):
    print(f"🎙️ Measuring ambient noise for {seconds} seconds...")
    duration_samples = int(seconds * samplerate)
    recording = sd.rec(duration_samples, samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    float_audio = recording.flatten().astype(np.float32) / 32768.0
    avg_volume = np.mean(np.abs(float_audio))
    print(f"🔈 Ambient noise level: {avg_volume:.5f}")
    return avg_volume

stream = sd.InputStream(device=device_index, samplerate=samplerate, channels=1, dtype='int16', blocksize=int(block_duration * samplerate), callback=audio_callback)

with stream:
    buffer = np.empty((0,), dtype=np.float32)

    ##########
    ambient_volume = measure_ambient_noise()
    volume_threshold = ambient_volume * 2.5    
    no_speech_threshold = 0.6 if ambient_volume < 0.01 else 0.75

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

                #handle_command(text)

                if not text:
                    continue

                if np.max(np.abs(segment_audio)) < volume_threshold:
                    print("Volume below dynamic threshold.")
                    continue
                if segment.no_speech_prob > 0.6:
                    continue
                if len(segment.text.strip().split()) <= 1:
                    continue
                if segment.end - segment.start < 0.5:
                    continue
                if np.max(np.abs(segment_audio)) < 0.01:
                    continue
                
                

                if key_word in text or is_similar_command(text, key_word):
                    print("Key word detected")
                    handle_command(text)
                
                intent, confidence = get_intent(text)
                if confidence > 0.6:
                    print(f"🤖 Intent: {intent} (confianza: {confidence:.2f})")
                    parse_command(intent,text)
                    wake_override = False


                
