import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
import time

@dataclass
class PipelineConfig:
    target_sr: int = 16000
    frame_length: int = 512
    vad_threshold: float = 0.5
    denoise_backend: Optional[str] = None
    denoise_model_path: Optional[Path] = None
    peak_norm_target_dbfs: float = -3.0

class RTFTracker:
    def __init__(self):
        self.total_audio_s = 0.0
        self.total_proc_s = 0.0
        self.rtf = 0.0
    
    def update(self, audio_duration_s: float, processing_s: float):
        self.total_audio_s += audio_duration_s
        self.total_proc_s += processing_s
        if self.total_audio_s > 0:
            self.rtf = self.total_proc_s / self.total_audio_s
    
    def reset(self):
        self.total_audio_s = 0.0
        self.total_proc_s = 0.0
        self.rtf = 0.0
    
    def summary(self) -> Dict[str, float]:
        return {"rtf": self.rtf, "total_audio_s": self.total_audio_s, "total_proc_s": self.total_proc_s}

def peak_normalise(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    if target_dbfs > 0:
        raise ValueError("target_dbfs must be <= 0")
    peak = np.max(np.abs(audio))
    if peak > 0:
        gain = 10 ** (target_dbfs / 20) / peak
        return audio * gain
    return audio

class SileroVAD:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._loaded = False
    
    def _load_model(self):
        self._loaded = True
    
    def is_speech(self, frame: np.ndarray) -> bool:
        if not self._loaded:
            self._load_model()
        rms = np.sqrt(np.mean(frame**2))
        return rms > 0.01

class NeuralDenoiser:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
    
    def denoise(self, frame: np.ndarray) -> np.ndarray:
        return frame

class AudioMLOptimizer:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.vad = SileroVAD(threshold=self.config.vad_threshold)
        self.denoiser = NeuralDenoiser(self.config.denoise_model_path)
        self.tracker = RTFTracker()
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None, show_progress: bool = True) -> np.ndarray:
        import soundfile as sf
        audio, sr = sf.read(input_path)
        return self.process_array(audio, sr)
    
    def process_array(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Traite un tableau audio directement (pour Streamlit)"""
        from scipy import signal
        
        if orig_sr != self.config.target_sr:
            audio = signal.resample_poly(audio, self.config.target_sr, orig_sr)
        
        frame_len = self.config.frame_length
        hop = frame_len // 2
        processed_frames = []
        start_time = time.time()
        
        for i in range(0, len(audio) - frame_len + 1, hop):
            frame = audio[i:i+frame_len].astype(np.float32)
            is_speech = self.vad.is_speech(frame)
            if not is_speech:
                continue
            clean_frame = self.denoiser.denoise(frame)
            processed_frames.append(clean_frame)
        
        if not processed_frames:
            return audio
        
        output = np.concatenate(processed_frames)
        output = peak_normalise(output, self.config.peak_norm_target_dbfs)
        self.tracker.update(len(audio) / self.config.target_sr, time.time() - start_time)
        return output
    
    def stream(self, audio: np.ndarray, orig_sr: int) -> Iterator[np.ndarray]:
        if orig_sr != self.config.target_sr:
            from scipy import signal
            audio = signal.resample_poly(audio, self.config.target_sr, orig_sr)
        
        frame_len = self.config.frame_length
        hop = frame_len // 2
        
        for i in range(0, len(audio) - frame_len + 1, hop):
            frame = audio[i:i+frame_len].astype(np.float32)
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                clean = self.denoiser.denoise(frame)
                yield clean
    
    def get_rtf_report(self) -> Dict[str, float]:
        return self.tracker.summary()
