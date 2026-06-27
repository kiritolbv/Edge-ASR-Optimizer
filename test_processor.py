"""
Unit tests for Edge ASR Optimizer.

Tests cover the pure-Python DSP utilities (no ML model required)
and the pipeline integration with mocked VAD / Denoiser.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from edge_asr_optimizer.processor import (
    AudioMLOptimizer,
    PipelineConfig,
    RTFTracker,
    _spectral_subtraction,
    peak_normalise,
    resample,
)


# ───────────────────────────── fixtures ────────────────────────────────────

@pytest.fixture
def white_noise() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(16_000).astype(np.float32) * 0.1


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(vad_threshold=0.5, denoise_model_path=None)


# ───────────────────────────── resample ────────────────────────────────────

class TestResample:
    def test_noop_same_rate(self, white_noise: np.ndarray) -> None:
        out = resample(white_noise, 16_000, 16_000)
        np.testing.assert_array_equal(out, white_noise)

    def test_downsample_length(self, white_noise: np.ndarray) -> None:
        out = resample(white_noise, 48_000, 16_000)
        expected = len(white_noise) * 16_000 // 48_000
        assert abs(len(out) - expected) <= 2  # polyphase may differ by 1–2

    def test_output_dtype(self, white_noise: np.ndarray) -> None:
        out = resample(white_noise, 44_100, 16_000)
        assert out.dtype == np.float32


# ───────────────────────────── peak_normalise ──────────────────────────────

class TestPeakNormalise:
    def test_target_peak_minus3(self, white_noise: np.ndarray) -> None:
        out = peak_normalise(white_noise, target_dbfs=-3.0)
        peak_db = 20 * np.log10(np.max(np.abs(out)))
        assert abs(peak_db - (-3.0)) < 0.01

    def test_silent_passthrough(self) -> None:
        silence = np.zeros(320, dtype=np.float32)
        out = peak_normalise(silence)
        np.testing.assert_array_equal(out, silence)

    def test_positive_dbfs_raises(self, white_noise: np.ndarray) -> None:
        with pytest.raises(ValueError, match="target_dbfs must be <= 0"):
            peak_normalise(white_noise, target_dbfs=1.0)

    def test_output_dtype(self, white_noise: np.ndarray) -> None:
        assert peak_normalise(white_noise).dtype == np.float32


# ───────────────────────────── spectral subtraction ────────────────────────

class TestSpectralSubtraction:
    def test_same_length(self, white_noise: np.ndarray) -> None:
        frame = white_noise[:320]
        out = _spectral_subtraction(frame)
        assert len(out) == len(frame)

    def test_reduces_power(self, white_noise: np.ndarray) -> None:
        frame = white_noise[:320]
        out = _spectral_subtraction(frame)
        assert np.var(out) < np.var(frame)

    def test_output_dtype(self, white_noise: np.ndarray) -> None:
        assert _spectral_subtraction(white_noise[:320]).dtype == np.float32


# ───────────────────────────── RTFTracker ──────────────────────────────────

class TestRTFTracker:
    def test_initial_rtf_is_zero(self) -> None:
        assert RTFTracker().rtf == 0.0

    def test_update_and_rtf(self) -> None:
        t = RTFTracker()
        t.update(audio_duration_s=1.0, processing_s=0.05)
        assert abs(t.rtf - 0.05) < 1e-9

    def test_reset(self) -> None:
        t = RTFTracker()
        t.update(1.0, 0.1)
        t.reset()
        assert t.rtf == 0.0

    def test_summary_keys(self) -> None:
        t = RTFTracker()
        t.update(2.0, 0.08)
        assert set(t.summary().keys()) == {"rtf", "total_audio_s", "total_proc_s"}


# ───────────────────────────── AudioMLOptimizer ────────────────────────────

class TestAudioMLOptimizer:
    """Integration tests with mocked ML models."""

    @pytest.fixture
    def optimizer(self, config: PipelineConfig) -> AudioMLOptimizer:
        with (
            patch("edge_asr_optimizer.processor.SileroVAD._load_model"),
            patch.object(
                AudioMLOptimizer, "_init_models",
                lambda self: setattr(self, "vad", MagicMock(is_speech=lambda _: True))
                or setattr(self, "denoiser", MagicMock(denoise=lambda f: f)),
            ),
        ):
            return AudioMLOptimizer(config)

    def test_repr(self, optimizer: AudioMLOptimizer) -> None:
        r = repr(optimizer)
        assert "AudioMLOptimizer" in r
        assert "sr=16000" in r

    def test_process_frame_returns_array(
        self, optimizer: AudioMLOptimizer, white_noise: np.ndarray
    ) -> None:
        frame = white_noise[:320]
        optimizer.vad.is_speech = MagicMock(return_value=True)
        optimizer.denoiser.denoise = MagicMock(return_value=frame)
        result = optimizer.process_frame(frame)
        assert result is not None
        assert result.dtype == np.float32

    def test_process_frame_vad_suppression(
        self, optimizer: AudioMLOptimizer, white_noise: np.ndarray
    ) -> None:
        frame = white_noise[:320]
        optimizer.vad.is_speech = MagicMock(return_value=False)
        result = optimizer.process_frame(frame)
        assert result is None

    def test_stream_yields_frames(
        self, optimizer: AudioMLOptimizer, white_noise: np.ndarray
    ) -> None:
        optimizer.vad.is_speech = MagicMock(return_value=True)
        optimizer.denoiser.denoise = MagicMock(side_effect=lambda f: f)
        frames = list(optimizer.stream(white_noise, orig_sr=16_000))
        assert len(frames) > 0
        assert all(f.dtype == np.float32 for f in frames)

@pytest.mark.slow
def test_wer_improvement_on_synthetic_noise(tmp_path):
    """Requires a clean audio file and a reference."""
    from edge_asr_optimizer import AudioMLOptimizer, PipelineConfig
    import whisper
    from jiwer import wer
    import soundfile as sf
    import numpy as np

    # Generate a simple tone or skip if no real file
    # This is a placeholder – you need a real speech sample.
    pytest.skip("No reference speech file for WER test")
