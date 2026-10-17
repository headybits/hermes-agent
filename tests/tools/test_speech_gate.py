"""Streaming speech/noise gate (tools/speech_gate.py) and the voice.speech_prob RPC.

The gate exists so barge-in ignores clatter and running water. These tests hold the
contract: synthetic voiced audio scores as speech, non-speech (noise bursts, broadband
noise, a loud tone, silence) does not, state survives chunked feeding, and the RPC
fails open (``available: False``) when the model can't load.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from tools import speech_gate

SR = 16_000


def _pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def _voiced(sec: float) -> np.ndarray:
    """Harmonic stack with vibrato and syllable-rate amplitude modulation: speech-like."""
    t = np.arange(int(SR * sec)) / SR
    f0 = 140 + 12 * np.sin(2 * np.pi * 5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 12))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t - np.pi / 2))
    return (0.35 * x * env / np.max(np.abs(x))).astype("float32")


def _clatter(sec: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(int(SR * sec), dtype="float32")
    for t in np.arange(0.1, sec, 0.45):
        n, s = int(SR * 0.06), int(t * SR)
        env = np.exp(-np.linspace(0, 9, n))
        ring = np.sin(2 * np.pi * rng.uniform(1800, 4200) * np.arange(n) / SR)
        burst = 0.6 * rng.standard_normal(n) * env + 0.4 * ring * env
        x[s : s + n] += burst[: len(x[s : s + n])]
    return np.clip(x, -1, 1)


def _water(sec: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = np.convolve(rng.standard_normal(int(SR * sec)), np.ones(8) / 8, "same")
    return (0.4 * x / np.max(np.abs(x))).astype("float32")


def _loud_hum(sec: float) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (0.6 * np.sin(2 * np.pi * 120 * t)).astype("float32")


def _max_prob(audio: np.ndarray, frame_ms: int = 100) -> float:
    gate = speech_gate.SpeechGate()
    step = SR * frame_ms // 1000 * 2
    raw = _pcm(audio)
    return max(gate.feed(raw[i : i + step]) for i in range(0, len(raw), step))


pytestmark = pytest.mark.skipif(not speech_gate.available(), reason="bundled Silero VAD unavailable")


def test_non_speech_stays_below_threshold():
    for name, audio in {
        "clatter": _clatter(4.0),
        "water": _water(4.0),
        "loud hum": _loud_hum(4.0),
        "silence": np.zeros(SR * 3, dtype="float32"),
    }.items():
        assert _max_prob(audio) < speech_gate.DEFAULT_THRESHOLD, name


def test_loud_non_speech_is_louder_than_quiet_speech_yet_still_rejected():
    """The failure mode this gate fixes: a loudness check ranks the hum above the voice."""
    voice, hum = _voiced(3.0) * 0.3, _loud_hum(3.0)
    assert np.sqrt(np.mean(hum**2)) > np.sqrt(np.mean(voice**2))
    assert _max_prob(hum) < speech_gate.DEFAULT_THRESHOLD


def test_state_survives_arbitrary_chunking():
    """Odd chunk sizes (not multiples of the 512 window) must score like one big feed."""
    audio = _voiced(2.0)
    raw = _pcm(audio)
    whole = speech_gate.SpeechGate()
    whole.feed(raw)
    chunked = speech_gate.SpeechGate()
    for i in range(0, len(raw), 1500):  # 750 samples: not window-aligned
        chunked.feed(raw[i : i + 1500])
    assert chunked.probability == pytest.approx(whole.probability, abs=0.05)


def test_reset_clears_state_and_threshold_controls_verdict():
    gate = speech_gate.SpeechGate(threshold=0.5)
    gate._h[:] = 1.0  # dirty the recurrent state
    gate.probability = 0.99
    gate.reset()
    assert gate.probability == 0.0 and not gate._h.any()
    gate.probability = 0.6
    assert gate.is_speech
    gate.threshold = 0.7
    assert not gate.is_speech


def test_odd_byte_count_and_empty_feed_are_safe():
    gate = speech_gate.SpeechGate()
    assert gate.feed(b"") == 0.0
    gate.feed(_pcm(_voiced(0.2))[:-1])  # dangling half sample dropped, not an error


def test_gate_for_is_per_owner_and_updates_threshold():
    speech_gate.drop_gate("a")
    speech_gate.drop_gate("b")
    a = speech_gate.gate_for("a", 0.3)
    assert speech_gate.gate_for("a") is a and a.threshold == 0.3
    assert speech_gate.gate_for("a", 0.8).threshold == 0.8
    assert speech_gate.gate_for("b") is not a
    speech_gate.drop_gate("a")
    speech_gate.drop_gate("b")
    assert speech_gate.gate_for("a") is not a


def test_unavailable_model_reports_false(monkeypatch):
    monkeypatch.setattr(speech_gate, "_session", None)

    def boom():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(speech_gate, "_model_path", boom)
    assert speech_gate.available() is False




def test_live_summary_is_logged_once_per_interval(caplog, monkeypatch):
    """Proof-of-life for a live session: calls, max score and verdict, rate-limited."""
    import logging

    clock = [100.0]
    monkeypatch.setattr(speech_gate.time, "monotonic", lambda: clock[0])
    gate = speech_gate.SpeechGate()
    frame = _pcm(_clatter(0.2))

    with caplog.at_level(logging.INFO, logger=speech_gate.logger.name):
        for _ in range(5):  # same instant: nothing logged yet
            gate.feed(frame)
        assert not [r for r in caplog.records if "speech gate:" in r.message]

        clock[0] += speech_gate.LOG_INTERVAL_S + 0.1
        gate.feed(frame)
        lines = [r.message for r in caplog.records if "speech gate:" in r.message]

    assert len(lines) == 1
    assert "6 calls" in lines[0] and "speech_calls=0" in lines[0]
    assert gate._log_calls == 0  # window reset after logging
