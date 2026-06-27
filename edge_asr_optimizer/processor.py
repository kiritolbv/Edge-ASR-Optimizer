"""
Edge ASR Optimizer — Core Processing Module.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional
import numpy as np
import scipy.signal as sps
import soundfile as sf

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    target_sr: int = 16_000
    frame_duration_ms: int = 20
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    denoise_backend: str = "onnx"
    denoise_model_path: Optional[Path] = None
    peak_norm_target_dbfs: float = -3.0
    device: str = "cpu"
    log_level: int = logging.INFO
    frame_length: int = field(init=False)
    hop_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.frame_length = int(self.target_sr * self.frame_duration_ms / 1000)
        self.hop_length = self.frame_length // 2
        logging.basicConfig(
            format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            datefmt="%H:%M:%S",
            level=self.log_level,
        )

@dataclass
class RTFTracker:
    _total_audio_s: float = field(default=0.0, init=False)
    _total_proc_s: float = field(default=0.0, init=False)

    @property
    def rtf(self) -> float:
        if self._total_audio_s == 0.0:
            return 0.0
        return self._total_proc_s / self._total_audio_s

    def update(self, audio_duration_s: float, processing_s: float) -> None:
        self._total_audio_s += audio_duration_s
        self._total_proc_s += processing_s

    def reset(self) -> None:
        self._total_audio_s = 0.0
        self._total_proc_s = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "rtf": round(self.rtf, 4),
            "total_audio_s": round(self._total_audio_s, 4),
            "total_proc_s": round(self._total_proc_s, 6),
        }

class SileroVAD:
    _model = None

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16_000) -> None:
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self._load_model()
        logger.info("SileroVAD initialised (threshold=%.2f)", threshold)

    def _load_model(self) -> None:
        try:
            import torch
            if SileroVAD._model is None:
                logger.debug("Loading Silero VAD from torch.hub …")
                SileroVAD._model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                SileroVAD._model.eval()
                logger.info("Silero VAD loaded successfully")
            self._model_instance = SileroVAD._model
            self._torch = torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for SileroVAD. Install it with: pip install torch torchaudio"
            ) from exc

    def probability(self, frame: np.ndarray) -> float:
        tensor = self._torch.from_numpy(frame).unsqueeze(0)
        with self._torch.no_grad():
            prob: float = self._model_instance(tensor, self.sampling_rate).item()
        return prob

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= self.threshold

class NeuralDenoiser:
    def __init__(self, model_path: Optional[Path] = None, backend: str = "onnx", device: str = "cpu") -> None:
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self._session = None
        self._torch_model = None
        if model_path is not None:
            self._load_model()

    def _load_model(self) -> None:
        if self.backend == "onnx":
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            except ImportError as exc:
                raise ImportError("onnxruntime is required for ONNX backend. Install with: pip install onnxruntime") from exc
        elif self.backend == "torch":
            try:
                import torch
                self._torch_model = torch.jit.load(str(self.model_path), map_location=self.device)
                self._torch_model.eval()
                self._torch = torch
            except ImportError as exc:
                raise ImportError("torch is required for the torch backend.") from exc
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Choose 'onnx' or 'torch'.")

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: frame[np.newaxis, np.newaxis, :].astype(np.float32)})
            return output[0].squeeze().astype(np.float32)
        if self._torch_model is not None:
            tensor = self._torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
            with self._torch.no_grad():
                out = self._torch_model(tensor)
            return out.squeeze().numpy().astype(np.float32)
        return _spectral_subtraction(frame)

def _spectral_subtraction(frame: np.ndarray, alpha: float = 2.0, beta: float = 0.001) -> np.ndarray:
    n_fft = max(512, len(frame))
    spectrum = np.fft.rfft(frame, n=n_fft)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    noise_estimate = np.percentile(magnitude, 25)
    magnitude_clean = np.maximum(magnitude - alpha * noise_estimate, beta * noise_estimate)
    clean_spectrum = magnitude_clean * np.exp(1j * phase)
    clean = np.fft.irfft(clean_spectrum, n=n_fft)[: len(frame)]
    return clean.astype(np.float32)

def peak_normalise(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    if target_dbfs > 0:
        raise ValueError(f"target_dbfs must be <= 0, got {target_dbfs}")
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_dbfs / 20.0)
    return (audio * (target_linear / peak)).astype(np.float32)

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    resampled = sps.resample_poly(audio, up, down)
    return resampled.astype(np.float32)

class AudioMLOptimizer:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.rtf_tracker = RTFTracker()
        self._init_models()
        logger.info("AudioMLOptimizer ready | sr=%d Hz | frame=%d ms", self.config.target_sr, self.config.frame_duration_ms)

    def _init_models(self) -> None:
        self.vad = SileroVAD(threshold=self.config.vad_threshold)
        self.denoiser = NeuralDenoiser(
            model_path=self.config.denoise_model_path,
            backend=self.config.denoise_backend,
            device=self.config.device,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        t0 = time.perf_counter()
        audio_s = len(frame) / self.config.target_sr
        vad_frame = frame[:256] if len(frame) >= 256 else frame
        if False:  # VAD désactivé temporairement
            self.rtf_tracker.update(audio_s, time.perf_counter() - t0)
            return None
        denoised = self.denoiser.denoise(frame)
        normalised = peak_normalise(denoised, self.config.peak_norm_target_dbfs)
        self.rtf_tracker.update(audio_s, time.perf_counter() - t0)
        return normalised

    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        logger.info("Processing file: %s", input_path)
        audio, orig_sr = sf.read(str(input_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        audio = resample(audio, orig_sr, self.config.target_sr)
        processed_frames: list[np.ndarray] = []
        fl = self.config.frame_length
        for start in range(0, len(audio), fl):
            frame = audio[start:start+fl]
            if len(frame) < fl:
                frame = np.pad(frame, (0, fl - len(frame)))
            result = self.process_frame(frame)
            if result is not None:
                processed_frames.append(result)
            else:
                processed_frames.append(np.zeros(fl, dtype=np.float32))
        clean_audio = np.concatenate(processed_frames) if processed_frames else np.array([], dtype=np.float32)
        if output_path is not None:
            sf.write(str(output_path), clean_audio, self.config.target_sr, subtype="PCM_16")
            logger.info("Saved processed audio → %s", output_path)
        return clean_audio

    def stream(self, audio: np.ndarray, orig_sr: int, yield_silence: bool = False) -> Generator[np.ndarray, None, None]:
        audio_16k = resample(audio, orig_sr, self.config.target_sr)
        fl = self.config.frame_length
        for start in range(0, len(audio_16k) - fl + 1, fl):
            frame = audio_16k[start:start+fl]
            result = self.process_frame(frame)
            if result is not None:
                yield result
            elif yield_silence:
                yield np.zeros(fl, dtype=np.float32)

    def reset(self) -> None:
        self.rtf_tracker.reset()

    def __repr__(self) -> str:
        return f"AudioMLOptimizer(sr={self.config.target_sr}, frame_ms={self.config.frame_duration_ms}, vad_threshold={self.config.vad_threshold}, device={self.config.device})"

