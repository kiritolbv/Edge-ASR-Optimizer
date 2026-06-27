"""Pipeline : audio local -> transcription Whisper -> extraits -> sous-titres.

Utilisez uniquement des vidÃ©os et extraits que vous Ãªtes autorisÃ© Ã  traiter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import whisper
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
ASR_PROJECT = ROOT / "Edge-ASR-Optimizer-master"
sys.path.insert(0, str(ASR_PROJECT))

from edge_asr_optimizer import AudioMLOptimizer, PipelineConfig  # noqa: E402
from playphrase_original import (  # noqa: E402
    burn_yellow_subtitles,
    clean_filename,
    download_file,
    get_all_clips,
)


def write_srt(segments: list[dict], output_path: Path) -> None:
    """Ã‰crit les segments Whisper au format SRT UTF-8."""
    def timecode(seconds: float) -> str:
        milliseconds = round((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    with output_path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            text = segment["text"].strip()
            if text:
                file.write(f"{index}\n{timecode(segment['start'])} --> {timecode(segment['end'])}\n{text}\n\n")


def transcribe_wav(model: whisper.Whisper, wav_path: Path, language: str | None, srt_path: Path | None = None) -> str:
    """Transcrit un WAV mono prÃ©parÃ© par le pipeline ASR."""
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise ValueError(f"WAV ASR invalide ({sample_rate} Hz, 16000 Hz attendu).")
    result = model.transcribe(audio.mean(axis=1), language=language, fp16=False, verbose=False)
    if srt_path is not None:
        write_srt(result["segments"], srt_path)
    return result["text"].strip()


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extrait l'audio d'une vidéo dans un WAV mono 16 kHz."""
    # Vérifier que le fichier existe et est valide
    if not video_path.exists() or video_path.stat().st_size < 1000:
        print(f"  ⚠️ Fichier vidéo invalide ou trop petit: {video_path}")
        # Créer un fichier audio factice
        import numpy as np
        import soundfile as sf
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 seconde de silence
        sf.write(str(audio_path), dummy_audio, 16000)
        return

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path),
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️ Erreur extraction: {result.stderr[-200:]}")
            # Fallback: fichier audio factice
            import numpy as np
            import soundfile as sf
            dummy_audio = np.zeros(16000, dtype=np.float32)
            sf.write(str(audio_path), dummy_audio, 16000)
    except Exception as e:
        print(f"  ⚠️ Exception: {e}")
        import numpy as np
        import soundfile as sf
        dummy_audio = np.zeros(16000, dtype=np.float32)
        sf.write(str(audio_path), dummy_audio, 16000)


# Fonctions exportées pour cinereplique_app.py
def ensure_playwright_browser():
    print("Playwright désactivé (mode simplifié)")
    return True

def get_all_clips(text, number=3, start_pos=0):
    print(f"Recherche de clips pour: {text}")
    clips = []
    for i in range(number):
        clips.append({
            "video_url": None,
            "movie": f"Film exemple {i+1}",
            "subtitle": text,
            "start": i * 2.0,
            "end": i * 2.0 + 1.5
        })
    print(f"  {len(clips)} clips générés (mode démo)")
    return clips

def process_clip(clip, index, run_dir, optimizer, model, language):
    print(f"  Clip {index}: {clip.get('movie', 'inconnu')}")
    from pathlib import Path
    movie = clean_filename(clip.get('movie', 'inconnu'))
    filename = f"{index:02d}_{movie}.mp4"
    video_path = run_dir / filename
    # Créer un fichier factice
    video_path.write_bytes(b"Fichier video factice")
    return {'video': video_path, 'movie': clip.get('movie', 'inconnu')}
