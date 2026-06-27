import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.pipeline import AudioMLOptimizer, PipelineConfig
import whisper

st.set_page_config(page_title="Edge ASR Optimizer", layout="wide")

st.title("🎤 Edge ASR Optimizer")
st.markdown("### Optimisation audio avant reconnaissance vocale")

DURATION = 5
SAMPLE_RATE = 16000
LANGUAGE = "fr"

@st.cache_resource
def load_models():
    try:
        optimizer = AudioMLOptimizer(PipelineConfig(
            target_sr=16000,
            vad_threshold=0.5,
            denoise_backend=None,
            denoise_model_path=None
        ))
        whisper_model = whisper.load_model("base")
        return optimizer, whisper_model
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        return None, None

optimizer, whisper_model = load_models()

if st.button("🎙️ Enregistrer et transcrire"):
    if optimizer is None or whisper_model is None:
        st.error("❌ Modèles non chargés")
    else:
        with st.spinner("🎙️ Enregistrement en cours..."):
            audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                          samplerate=SAMPLE_RATE, 
                          channels=1, 
                          dtype=np.float32)
            sd.wait()
            audio = audio.flatten()
        
        with st.spinner("🧹 Prétraitement audio..."):
            try:
                processed_audio = optimizer.process_array(audio, SAMPLE_RATE)
            except Exception as e:
                st.warning(f"⚠️ Prétraitement ignoré: {e}")
                processed_audio = audio
        
        with st.spinner("🧠 Whisper recherche votre phrase..."):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, processed_audio, SAMPLE_RATE)
                try:
                    result = whisper_model.transcribe(f.name, language=LANGUAGE)
                    text = result["text"].strip()
                except Exception as e:
                    text = f"Erreur: {e}"
                finally:
                    if os.path.exists(f.name):
                        os.unlink(f.name)
        
        # AFFICHAGE PERSONNALISÉ
        if text and text != "":
            st.success(f"📝 **Whisper recherche votre phrase :** {text}")
        else:
            st.warning("⚠️ Aucune parole détectée. Réessayez.")
        
        with st.expander("🎵 Écouter l'audio"):
            st.audio(audio, sample_rate=SAMPLE_RATE)

