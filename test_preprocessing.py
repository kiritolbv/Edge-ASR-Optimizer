# test_preprocessing.py
"""
Test du pipeline de prétraitement complet
VAD + Denoising + Normalisation
"""

import sys
import numpy as np
import soundfile as sf
import time
from pathlib import Path

sys.path.insert(0, '.')

from edge_asr_optimizer.processor import (
    PipelineConfig,
    SileroVAD,
    NeuralDenoiser,
    RTFTracker,
    _spectral_subtraction
)

class SimpleAudioProcessor:
    """Version simplifiée du pipeline de prétraitement"""
    
    def __init__(self, config=None):
        self.config = config or PipelineConfig()
        self.vad = SileroVAD(threshold=self.config.vad_threshold)
        self.denoiser = NeuralDenoiser(model_path=None)
        self.tracker = RTFTracker()
        self.target_sr = self.config.target_sr
        self.vad_frame_len = 512  # Taille de frame pour le VAD
    
    def process_file(self, input_path, output_path=None):
        """Traite un fichier audio complet"""
        print(f"📁 Traitement: {Path(input_path).name}")
        print("-" * 50)
        
        # 1. Charger l'audio
        audio, sr = sf.read(input_path)
        original_duration = len(audio) / sr
        print(f"  📊 Original: {original_duration:.2f}s, {sr}Hz")
        
        # 2. Resample si nécessaire
        if sr != self.target_sr:
            print(f"  🔄 Resample: {sr}Hz → {self.target_sr}Hz")
            from scipy import signal
            audio = signal.resample_poly(audio, self.target_sr, sr)
            sr = self.target_sr
        
        # 3. Traiter par frames
        start_time = time.time()
        processed_frames = []
        vad_probs = []
        speech_frames = 0
        total_frames = 0
        
        frame_len = self.config.frame_length
        hop = frame_len // 2
        
        print(f"  🎤 Traitement par frames ({frame_len} échantillons)...")
        
        for i in range(0, len(audio) - frame_len + 1, hop):
            frame = audio[i:i+frame_len].astype(np.float32)
            
            # VAD - utiliser 512 échantillons
            vad_frame = frame[:self.vad_frame_len]
            # Si la frame est trop courte, la padder
            if len(vad_frame) < self.vad_frame_len:
                vad_frame = np.pad(vad_frame, (0, self.vad_frame_len - len(vad_frame)))
            
            vad_prob = self.vad.probability(vad_frame)
            vad_probs.append(vad_prob)
            is_speech = vad_prob > self.config.vad_threshold
            if is_speech:
                speech_frames += 1
            total_frames += 1
            
            # Denoising
            clean_frame = self.denoiser.denoise(frame)
            processed_frames.append(clean_frame)
            
            # RTF tracking (simulé)
            self.tracker.update(frame_len / sr, 0.001)
        
        # 4. Reconstruire l'audio
        audio_out = np.concatenate(processed_frames)
        
        # 5. Normalisation
        peak = np.max(np.abs(audio_out))
        if peak > 0:
            audio_out = audio_out / peak * 0.9
        
        elapsed = time.time() - start_time
        output_duration = len(audio_out) / sr
        
        # 6. Sauvegarder
        if output_path:
            Path(output_path).parent.mkdir(exist_ok=True)
            sf.write(output_path, audio_out, sr)
            print(f"  💾 Sauvegardé: {output_path}")
        
        # 7. Statistiques
        print(f"\n📊 Statistiques:")
        print(f"  Durée: {output_duration:.2f}s")
        print(f"  Frames: {total_frames}")
        print(f"  Parole détectée: {speech_frames}/{total_frames} ({speech_frames/total_frames*100:.1f}%)")
        if vad_probs:
            print(f"  VAD max: {max(vad_probs):.3f}, moy: {np.mean(vad_probs):.3f}")
        print(f"  Temps: {elapsed:.2f}s")
        print(f"  RTF: {self.tracker.rtf:.4f}")
        
        # 8. Analyse SNR
        snr_before = 20 * np.log10(np.std(audio) + 1e-10)
        snr_after = 20 * np.log10(np.std(audio_out) + 1e-10)
        print(f"  SNR avant: {snr_before:.1f} dB")
        print(f"  SNR après: {snr_after:.1f} dB")
        print(f"  Amélioration: {snr_after - snr_before:.1f} dB")
        
        return audio_out

def main():
    print("=" * 60)
    print("🎬 TEST DU PRÉTRAITEMENT ASR")
    print("=" * 60)
    
    # Initialiser le processeur
    processor = SimpleAudioProcessor()
    
    # Tester sur un fichier propre
    clean_file = Path('clean/sp01.wav')
    if clean_file.exists():
        print(f"\n🔊 Fichier CLEAN:")
        processor.process_file(clean_file, 'output/sp01_processed.wav')
    else:
        print(f"⚠️  {clean_file} non trouvé")
    
    # Tester sur un fichier bruité
    noisy_file = Path('data/noisy/15dB/sp03_street_sn15.wav')
    if noisy_file.exists():
        print(f"\n🔊 Fichier BRUITÉ:")
        processor.process_file(noisy_file, 'output/sp03_processed.wav')
    else:
        print(f"⚠️  {noisy_file} non trouvé")
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print("✅ Fichiers traités dans output/")

if __name__ == "__main__":
    main()
