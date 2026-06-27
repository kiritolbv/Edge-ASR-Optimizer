import whisper
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import os

print("=" * 60)
print("🎤 Test Whisper - Reconnaissance vocale")
print("=" * 60)

print("\n🎙️ Enregistrement de 5 secondes...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype=np.float32)
sd.wait()
audio = audio.flatten()

print("🧠 Whisper recherche votre phrase...")
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio, 16000)
    model = whisper.load_model("base")
    result = model.transcribe(f.name, language="fr")
    os.unlink(f.name)

print("\n" + "=" * 60)
print(f"📝 Whisper recherche votre phrase : {result['text']}")
print("=" * 60)
