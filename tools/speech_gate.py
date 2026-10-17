"""Streaming speech/noise gate for voice barge-in.

The desktop's barge-in monitor trips on loudness alone, so a pan clank or running
water interrupts the reply. This gate scores 16 kHz mono PCM with the Silero VAD
model that ships inside faster-whisper (no download) and answers one question:
is this audio human speech right now?

It is stateful: the model carries recurrent state between 512-sample windows, so
feed frames in order. One ``SpeechGate`` per audio stream.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # numpy / faster-whisper ship in the optional ``voice`` extras
    import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW = 512  # samples per Silero window at 16 kHz (32 ms)
CONTEXT = 64  # trailing samples of the previous window the model expects prepended
DEFAULT_THRESHOLD = 0.5
LOG_INTERVAL_S = 1.0  # one INFO summary per stream per second while audio flows

_session_lock = threading.Lock()
_session = None


def _model_path() -> Path:
    """Locate the Silero model that ships inside faster-whisper WITHOUT importing it.

    Importing faster_whisper loads ctranslate2's native libraries, which segfault at interpreter
    exit in a process that also hosts the gateway server; the gate only needs the ONNX file."""
    import importlib.util

    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.origin:
        raise FileNotFoundError("faster-whisper (bundled Silero VAD) is not installed")
    return Path(spec.origin).parent / "assets" / "silero_vad_v6.onnx"


def _get_session():
    """Process-wide ONNX session (stateless; per-stream state lives in SpeechGate).

    Raises ImportError/FileNotFoundError when the ``voice`` extras (faster-whisper's bundled
    model + onnxruntime) are absent; callers go through ``available()`` first."""
    global _session
    with _session_lock:
        if _session is None:
            import onnxruntime

            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 4
            _session = onnxruntime.InferenceSession(
                str(_model_path()), providers=["CPUExecutionProvider"], sess_options=opts
            )
        return _session


def available() -> bool:
    """True when the bundled Silero model and onnxruntime can be loaded."""
    try:
        import numpy

        _get_session()
        return True
    except (ImportError, OSError, ValueError) as exc:  # missing extra/asset, or an ONNX load failure
        logger.debug("speech gate unavailable: %s", exc)
        return False


class SpeechGate:
    """Per-stream Silero VAD. ``feed()`` takes int16 PCM bytes, returns the latest probability."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        import numpy as np

        self.threshold = float(threshold)
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros((1, CONTEXT), dtype="float32")
        self._buffer = np.zeros(0, dtype="float32")
        self.probability = 0.0
        self._log_started = time.monotonic()
        self._log_calls = 0
        self._log_max = 0.0
        self._log_speech_calls = 0

    def reset(self) -> None:
        self.__init__(self.threshold)

    def feed(self, pcm_int16: bytes) -> float:
        """Consume PCM (int16 LE, 16 kHz, mono). Partial windows are buffered across calls."""
        if not pcm_int16:
            return self.probability
        import numpy as np

        samples = np.frombuffer(pcm_int16[: len(pcm_int16) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
        self._buffer = np.concatenate([self._buffer, samples])
        session = _get_session()
        while len(self._buffer) >= WINDOW:
            window = self._buffer[:WINDOW].reshape(1, WINDOW)
            self._buffer = self._buffer[WINDOW:]
            model_in = np.concatenate([self._context, window], axis=1)
            out, self._h, self._c = session.run(None, {"input": model_in, "h": self._h, "c": self._c})
            self._context = window[:, -CONTEXT:]
            self.probability = float(out.reshape(-1)[-1])
            self._log_max = max(self._log_max, self.probability)
        self._log_calls += 1
        self._log_speech_calls += self.is_speech
        self._log_summary()
        return self.probability

    def _log_summary(self) -> None:
        """Rate-limited INFO line so a live session shows what the gate decides per second."""
        now = time.monotonic()
        if now - self._log_started < LOG_INTERVAL_S:
            return
        logger.info(
            "speech gate: %d calls/%.1fs, max_prob=%.2f, last_prob=%.2f, speech_calls=%d, threshold=%.2f",
            self._log_calls, now - self._log_started, self._log_max, self.probability,
            self._log_speech_calls, self.threshold,
        )
        self._log_started, self._log_calls, self._log_max, self._log_speech_calls = now, 0, 0.0, 0

    @property
    def is_speech(self) -> bool:
        return self.probability >= self.threshold


_gates: dict[str, SpeechGate] = {}
_gates_lock = threading.Lock()


def gate_for(owner: str, threshold: Optional[float] = None) -> SpeechGate:
    """One long-lived gate per connection/owner id so state survives between feed calls."""
    with _gates_lock:
        gate = _gates.get(owner)
        if gate is None:
            gate = _gates[owner] = SpeechGate(DEFAULT_THRESHOLD if threshold is None else threshold)
        elif threshold is not None:
            gate.threshold = float(threshold)
        return gate


def drop_gate(owner: str) -> None:
    with _gates_lock:
        _gates.pop(owner, None)
