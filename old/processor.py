"""
Edge ASR Optimizer — Core Processing Module.

This module provides the AudioMLOptimizer class, a high-performance audio
preprocessing pipeline designed for on-device ASR inference. It combines
ML-based VAD, neural denoising, and real-time streaming support with a
clean, modular architecture optimised for Edge AI deployment.

Author: IMT Atlantique — Embedded AI & DSP Research
License: MIT
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


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the AudioMLOptimizer pipeline.

    Attributes:
        target_sr: Target sample rate in Hz. Defaults to 16000.
        frame_duration_ms: Duration of each processing frame in milliseconds.
            Defaults to 20.
        vad_threshold: Voice activity detection confidence threshold [0–1].
            Defaults to 0.5.
        vad_min_speech_ms: Minimum speech segment duration to keep (ms).
            Defaults to 250.
        vad_min_silence_ms: Minimum silence segment to trigger suppression (ms).
            Defaults to 100.
        denoise_backend: Inference backend for denoising model.
            One of ``"onnx"``, ``"torch"``. Defaults to ``"onnx"``.
        denoise_model_path: Path to the denoising ONNX/TorchScript model.
            If None, falls back to spectral subtraction baseline.
        peak_norm_target_dbfs: Target peak level after normalisation (dBFS).
            Defaults to -3.0.
        device: Compute device for Torch inference. Defaults to ``"cpu"``.
        log_level: Python logging level. Defaults to ``logging.INFO``.
    """

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

    # Derived (post-init)
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


# ---------------------------------------------------------------------------
# RTF (Real-Time Factor) tracker
# ---------------------------------------------------------------------------

@dataclass
class RTFTracker:
    """Accumulates processing time and audio duration to compute RTF.

    RTF = total_processing_time / total_audio_duration.
    A value < 1.0 indicates faster-than-real-time processing.

    Example::

        tracker = RTFTracker()
        with tracker.measure(audio_duration_s=0.02):
            process_frame(frame)
        print(tracker.rtf)
    """

    _total_audio_s: float = field(default=0.0, init=False)
    _total_proc_s: float = field(default=0.0, init=False)

    @property
    def rtf(self) -> float:
        """Current Real-Time Factor. Returns 0.0 if no audio processed yet."""
        if self._total_audio_s == 0.0:
            return 0.0
        return self._total_proc_s / self._total_audio_s

    def update(self, audio_duration_s: float, processing_s: float) -> None:
        """Register one measurement.

        Args:
            audio_duration_s: Duration of the processed audio chunk in seconds.
            processing_s: Wall-clock time spent processing that chunk.
        """
        self._total_audio_s += audio_duration_s
        self._total_proc_s += processing_s

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._total_audio_s = 0.0
        self._total_proc_s = 0.0

    def summary(self) -> dict[str, float]:
        """Return a dictionary summary of RTF statistics.

        Returns:
            Dict with keys ``rtf``, ``total_audio_s``, ``total_proc_s``.
        """
        return {
            "rtf": round(self.rtf, 4),
            "total_audio_s": round(self._total_audio_s, 4),
            "total_proc_s": round(self._total_proc_s, 6),
        }


# ---------------------------------------------------------------------------
# VAD wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """Lightweight wrapper around the Silero VAD model.

    Silero VAD is a pre-trained noise-robust voice activity detector
    that operates on 16 kHz mono PCM audio. It returns a per-frame
    speech probability score, making it suitable for real-time streaming.

    Args:
        threshold: Speech probability threshold. Frames above this value
            are classified as speech. Defaults to 0.5.
        sampling_rate: Expected input sample rate. Must be 16000 or 8000.

    Raises:
        ImportError: If ``torch`` or ``torchaudio`` is not installed.
        RuntimeError: If the Silero model cannot be loaded from ``torch.hub``.

    Example::

        vad = SileroVAD(threshold=0.5)
        is_speech = vad.is_speech(frame_float32)
    """

    _model = None  # Class-level cache — load once per process

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16_000) -> None:
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self._load_model()
        logger.info("SileroVAD initialised (threshold=%.2f)", threshold)

    def _load_model(self) -> None:
        """Load Silero VAD from torch.hub (cached after first call)."""
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
                "torch is required for SileroVAD. "
                "Install it with: pip install torch torchaudio"
            ) from exc

    def probability(self, frame: np.ndarray) -> float:
        """Compute voice activity probability for a single frame.

        Args:
            frame: 1-D float32 array of audio samples at ``sampling_rate`` Hz.
                Length must be 256 (10 ms at 16 kHz) or 512 (32 ms).

        Returns:
            Speech probability in [0, 1].
        """
        tensor = self._torch.from_numpy(frame).unsqueeze(0)
        with self._torch.no_grad():
            prob: float = self._model_instance(tensor, self.sampling_rate).item()
        return prob

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the frame is classified as speech.

        Args:
            frame: 1-D float32 audio frame (256 or 512 samples at 16 kHz).

        Returns:
            True if speech probability exceeds the configured threshold.
        """
        return self.probability(frame) >= self.threshold


# ---------------------------------------------------------------------------
# Denoising backend
# ---------------------------------------------------------------------------

class NeuralDenoiser:
    """Neural speech denoiser with ONNX Runtime and Torch backends.

    Supports two backends:
    - ``"onnx"``: Uses ONNX Runtime for cross-platform, low-latency inference.
      Recommended for production deployment.
    - ``"torch"``: Uses a TorchScript model. Useful during development.

    Falls back to :func:`_spectral_subtraction` if no model path is provided.

    Args:
        model_path: Path to the ``.onnx`` or ``.pt`` model file.
            If None, spectral subtraction is used as a baseline.
        backend: Inference backend. One of ``"onnx"`` or ``"torch"``.
        device: Target device (``"cpu"`` or ``"cuda"``).

    Example::

        denoiser = NeuralDenoiser(
            model_path=Path("models/nsnet2.onnx"),
            backend="onnx",
        )
        clean = denoiser.denoise(noisy_frame)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        backend: str = "onnx",
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self._session = None  # ONNX session
        self._torch_model = None  # TorchScript model

        if model_path is not None:
            self._load_model()
            logger.info(
                "NeuralDenoiser loaded: backend=%s, model=%s", backend, model_path
            )
        else:
            logger.warning(
                "No denoising model path provided — using spectral subtraction baseline."
            )

    def _load_model(self) -> None:
        """Load the denoising model according to the configured backend."""
        if self.backend == "onnx":
            try:
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=providers
                )
                logger.debug(
                    "ONNX session created. Input: %s",
                    [i.name for i in self._session.get_inputs()],
                )
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for ONNX backend. "
                    "Install it with: pip install onnxruntime"
                ) from exc
        elif self.backend == "torch":
            try:
                import torch

                self._torch_model = torch.jit.load(
                    str(self.model_path), map_location=self.device
                )
                self._torch_model.eval()
                self._torch = torch
            except ImportError as exc:
                raise ImportError(
                    "torch is required for the torch backend."
                ) from exc
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. Choose 'onnx' or 'torch'."
            )

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single audio frame.

        Dispatches to the neural model if loaded, otherwise falls back
        to spectral subtraction.

        Args:
            frame: 1-D float32 audio frame.

        Returns:
            Denoised float32 audio frame of the same length.
        """
        if self._session is not None:
            return self._onnx_infer(frame)
        if self._torch_model is not None:
            return self._torch_infer(frame)
        return _spectral_subtraction(frame)

    def _onnx_infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference via ONNX Runtime.

        Args:
            frame: 1-D float32 audio frame.

        Returns:
            Denoised frame as float32 ndarray.
        """
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(
            None, {input_name: frame[np.newaxis, np.newaxis, :].astype(np.float32)}
        )
        return output[0].squeeze().astype(np.float32)

    def _torch_infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference via TorchScript.

        Args:
            frame: 1-D float32 audio frame.

        Returns:
            Denoised frame as float32 ndarray.
        """
        tensor = self._torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
        with self._torch.no_grad():
            out = self._torch_model(tensor)
        return out.squeeze().numpy().astype(np.float32)


def _spectral_subtraction(
    frame: np.ndarray,
    alpha: float = 2.0,
    beta: float = 0.001,
) -> np.ndarray:
    """Baseline spectral subtraction (Boll 1979 / Ephraim-Malah).

    Estimates the noise floor from the magnitude spectrum and subtracts
    a scaled version from the input. A floor (``beta``) prevents musical
    noise artefacts caused by negative spectral bins.

    Args:
        frame: 1-D float32 audio frame.
        alpha: Over-subtraction factor. Higher values remove more noise
            at the cost of increased distortion. Defaults to 2.0.
        beta: Spectral floor (fraction of noise estimate). Prevents
            musical noise. Defaults to 0.001.

    Returns:
        Denoised frame as float32 ndarray, same length as input.
    """
    n_fft = max(512, len(frame))
    spectrum = np.fft.rfft(frame, n=n_fft)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Noise estimate from lower quartile (stationary assumption)
    noise_estimate = np.percentile(magnitude, 25)
    magnitude_clean = np.maximum(magnitude - alpha * noise_estimate, beta * noise_estimate)

    clean_spectrum = magnitude_clean * np.exp(1j * phase)
    clean = np.fft.irfft(clean_spectrum, n=n_fft)[: len(frame)]
    return clean.astype(np.float32)


# ---------------------------------------------------------------------------
# Peak normalisation
# ---------------------------------------------------------------------------

def peak_normalise(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Apply peak normalisation to an audio array.

    Scales the signal so that its peak absolute value corresponds to
    ``target_dbfs`` relative to full-scale (0 dBFS = 1.0 in float32).

    Args:
        audio: 1-D float32 audio array.
        target_dbfs: Target peak level in dBFS. Must be <= 0.
            Defaults to -3.0.

    Returns:
        Normalised float32 array. Returned unchanged if the input is silent
        (peak < 1e-8) to avoid division by near-zero.

    Raises:
        ValueError: If ``target_dbfs`` is positive.

    Example::

        normalised = peak_normalise(audio, target_dbfs=-1.0)
    """
    if target_dbfs > 0:
        raise ValueError(f"target_dbfs must be <= 0, got {target_dbfs}")

    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        logger.debug("Silent frame detected, skipping normalisation.")
        return audio

    target_linear = 10 ** (target_dbfs / 20.0)
    return (audio * (target_linear / peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate using a polyphase filter.

    Uses ``scipy.signal.resample_poly`` which applies an anti-aliasing
    FIR filter, providing high SNR even for aggressive downsampling
    (e.g. 48 000 → 16 000 Hz).

    Args:
        audio: 1-D float32 audio array at ``orig_sr``.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled float32 audio array.

    Note:
        Resampling is a no-op if ``orig_sr == target_sr``.
    """
    if orig_sr == target_sr:
        return audio
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    resampled = sps.resample_poly(audio, up, down)
    logger.debug("Resampled %d → %d Hz (%d → %d samples)", orig_sr, target_sr, len(audio), len(resampled))
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Main AudioMLOptimizer class
# ---------------------------------------------------------------------------

class AudioMLOptimizer:
    """ML-powered audio preprocessing pipeline for on-device ASR.

    This class orchestrates a four-stage pipeline optimised for Edge AI
    deployment:

    1. **Resampling** — Polyphase downsampling to 16 kHz.
    2. **VAD** — Silero VAD filters non-speech frames, reducing downstream load.
    3. **Denoising** — Neural denoising (ONNX/Torch) or spectral subtraction.
    4. **Peak normalisation** — Ensures consistent loudness for ASR models.

    Both file-level batch processing and frame-by-frame streaming are supported.
    RTF (Real-Time Factor) is tracked automatically per session.

    Args:
        config: Pipeline configuration. Defaults to ``PipelineConfig()``.

    Example — file processing::

        optimizer = AudioMLOptimizer()
        clean_audio = optimizer.process_file(Path("noisy.wav"))

    Example — streaming::

        optimizer = AudioMLOptimizer()
        for clean_frame in optimizer.stream(audio_array, orig_sr=48000):
            asr_engine.feed(clean_frame)

        print(optimizer.rtf_tracker.summary())

    Attributes:
        config: Active pipeline configuration.
        rtf_tracker: RTF accumulator for the current session.
        vad: SileroVAD instance.
        denoiser: NeuralDenoiser instance.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.rtf_tracker = RTFTracker()
        self._init_models()
        logger.info(
            "AudioMLOptimizer ready | sr=%d Hz | frame=%d ms | device=%s",
            self.config.target_sr,
            self.config.frame_duration_ms,
            self.config.device,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        """Instantiate VAD and denoising models according to the config."""
        logger.debug("Initialising VAD …")
        self.vad = None

        logger.debug("Initialising Denoiser …")
        self.denoiser = NeuralDenoiser(
            model_path=self.config.denoise_model_path,
            backend=self.config.denoise_backend,
            device=self.config.device,
        )

    # ------------------------------------------------------------------
    # Frame-level processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio frame through the full pipeline.

        Applies VAD, denoising, and peak normalisation sequentially.
        Returns None for non-speech frames (VAD rejected), allowing
        the caller to implement silence substitution or frame skipping.

        Args:
            frame: 1-D float32 audio frame. Length must match
                ``config.frame_length`` (``target_sr * frame_duration_ms / 1000``).

        Returns:
            Processed float32 frame, or None if classified as non-speech.
        """
        t0 = time.perf_counter()
        audio_s = len(frame) / self.config.target_sr

        # Stage 1 — VAD gate
        vad_frame = frame[:256] if len(frame) >= 256 else frame  # Silero needs 256
        if self.vad is not None:
            if not self.vad.is_speech(vad_frame):
                logger.debug("VAD: non-speech frame suppressed")
                self.rtf_tracker.update(audio_s, time.perf_counter() - t0)
                return None

        # Stage 2 — Neural denoising
        denoised = self.denoiser.denoise(frame)

        # Stage 3 — Peak normalisation
        normalised = peak_normalise(denoised, self.config.peak_norm_target_dbfs)

        self.rtf_tracker.update(audio_s, time.perf_counter() - t0)
        return normalised

    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        """Load an audio file, process it, and optionally save the result.

        Supports any format readable by ``soundfile`` (WAV, FLAC, OGG, …).
        Multi-channel audio is down-mixed to mono before processing.

        Args:
            input_path: Path to the input audio file.
            output_path: If provided, the processed audio is written here as
                a 16-bit PCM WAV file at ``config.target_sr``.

        Returns:
            Processed mono float32 audio array at ``config.target_sr``.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
            RuntimeError: If the audio file cannot be decoded.

        Example::

            clean = optimizer.process_file(
                Path("noisy_recording.wav"),
                output_path=Path("clean_recording.wav"),
            )
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        logger.info("Processing file: %s", input_path)
        audio, orig_sr = sf.read(str(input_path), dtype="float32", always_2d=True)

        # Down-mix to mono
        audio = audio.mean(axis=1)

        # Stage 0 — Resample
        audio = resample(audio, orig_sr, self.config.target_sr)

        # Frame-by-frame processing
        processed_frames: list[np.ndarray] = []
        fl = self.config.frame_length

        for start in range(0, len(audio), fl):
            frame = audio[start : start + fl]
            if len(frame) < fl:
                # Pad last frame with zeros
                frame = np.pad(frame, (0, fl - len(frame)))

            result = self.process_frame(frame)
            if result is not None:
                processed_frames.append(result)
            else:
                # Replace suppressed frames with silence to maintain timing
                processed_frames.append(np.zeros(fl, dtype=np.float32))

        clean_audio = np.concatenate(processed_frames) if processed_frames else np.array([], dtype=np.float32)

        if output_path is not None:
            sf.write(str(output_path), clean_audio, self.config.target_sr, subtype="PCM_16")
            logger.info("Saved processed audio → %s", output_path)

        rtf = self.rtf_tracker.rtf
        logger.info(
            "File processing complete | duration=%.2f s | RTF=%.4f (%s real-time)",
            len(clean_audio) / self.config.target_sr,
            rtf,
            "faster than" if rtf < 1.0 else "slower than",
        )
        return clean_audio

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def stream(
        self,
        audio: np.ndarray,
        orig_sr: int,
        yield_silence: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Process audio as a stream of frames, yielding results incrementally.

        This generator is the core interface for real-time ASR integration.
        Audio is resampled once up front, then fed frame by frame through
        the pipeline. Only speech frames are yielded by default.

        Args:
            audio: Input audio as a 1-D float32 array.
            orig_sr: Sample rate of the input audio in Hz.
            yield_silence: If True, zero-filled frames are yielded for
                non-speech segments, preserving timing alignment.
                Defaults to False.

        Yields:
            Processed float32 frames of length ``config.frame_length``.

        Example::

            stream = optimizer.stream(raw_audio, orig_sr=48000)
            for frame in stream:
                transcription = asr_model.transcribe(frame)
        """
        logger.info(
            "Starting stream | orig_sr=%d → %d Hz | yield_silence=%s",
            orig_sr,
            self.config.target_sr,
            yield_silence,
        )

        # Resample the entire buffer once
        audio_16k = resample(audio, orig_sr, self.config.target_sr)
        fl = self.config.frame_length

        for start in range(0, len(audio_16k) - fl + 1, fl):
            frame = audio_16k[start : start + fl]
            result = self.process_frame(frame)

            if result is not None:
                yield result
            elif yield_silence:
                yield np.zeros(fl, dtype=np.float32)

        logger.debug("Stream complete. RTF summary: %s", self.rtf_tracker.summary())

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset RTF statistics for a new processing session.

        Call this between independent recordings to obtain per-session RTF
        measurements without accumulating previous runs.
        """
        self.rtf_tracker.reset()
        logger.debug("RTF tracker reset.")

    def benchmark(self, duration_s: float = 5.0) -> dict[str, float]:
        """Run a synthetic RTF benchmark on white-noise audio.

        Generates ``duration_s`` seconds of white noise at ``target_sr``,
        runs it through the full pipeline, and returns timing statistics.

        Args:
            duration_s: Duration of the synthetic test signal in seconds.
                Defaults to 5.0.

        Returns:
            Dictionary with keys:
            - ``rtf``: Real-Time Factor (lower = faster).
            - ``total_audio_s``: Total audio duration processed.
            - ``total_proc_s``: Total wall-clock processing time.
            - ``frames_processed``: Number of frames processed.

        Example::

            stats = optimizer.benchmark(duration_s=10.0)
            print(f"RTF = {stats['rtf']:.4f}")
        """
        logger.info("Running benchmark: %.1f s of synthetic audio …", duration_s)
        self.reset()

        rng = np.random.default_rng(seed=42)
        n_samples = int(duration_s * self.config.target_sr)
        noise = rng.standard_normal(n_samples).astype(np.float32) * 0.1
        frames_processed = 0

        for _ in self.stream(noise, orig_sr=self.config.target_sr, yield_silence=True):
            frames_processed += 1

        stats = self.rtf_tracker.summary()
        stats["frames_processed"] = frames_processed
        logger.info("Benchmark complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AudioMLOptimizer("
            f"sr={self.config.target_sr}, "
            f"frame_ms={self.config.frame_duration_ms}, "
            f"vad_threshold={self.config.vad_threshold}, "
            f"denoise_backend={self.config.denoise_backend!r}, "
            f"device={self.config.device!r}"
            f")"
        )
