"""`voice.speech_prob` is how the desktop's barge-in monitor asks "is that sound human speech?".

The contract: PCM in, a speech probability out; per-caller state so chunked frames behave
like one stream; ``available: False`` (never an error) when the bundled VAD can't load, so
the client falls back to loudness-only barge-in instead of going deaf; bad input is a 4001.
"""

import base64

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from tools import speech_gate
from tui_gateway import server

SR = 16_000


def _b64(x):
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def _call(**params):
    return server._methods["voice.speech_prob"](1, params)


@pytest.fixture(autouse=True)
def _fresh_gates():
    speech_gate._gates.clear()
    yield
    speech_gate._gates.clear()


def _clatter():
    rng = np.random.default_rng(0)
    x = np.zeros(SR, dtype="float32")
    for s in range(1600, SR - 960, 7200):
        n = 960
        x[s : s + n] = rng.standard_normal(n) * np.exp(-np.linspace(0, 9, n)) * 0.6
    return x


def test_noise_scores_as_not_speech():
    answer = _call(pcm=_b64(_clatter()), sample_rate=SR, reset=True)["result"]

    assert answer["available"] is True
    assert answer["speech"] is False
    assert 0.0 <= answer["probability"] < 0.5


def test_threshold_param_moves_the_verdict_cutoff():
    _call(pcm=_b64(_clatter()), sample_rate=SR, reset=True)
    # any probability >= 0.0001 counts as speech at this threshold
    answer = _call(pcm=_b64(_clatter()), sample_rate=SR, threshold=0.0001)["result"]

    assert answer["speech"] is True


def test_reset_drops_stream_state():
    _call(pcm=_b64(_clatter()), sample_rate=SR)
    assert speech_gate._gates

    _call(reset=True)
    assert speech_gate._gates == {} or all(g.probability == 0.0 for g in speech_gate._gates.values())


def test_empty_pcm_is_not_an_error():
    answer = _call(reset=True)["result"]

    assert answer["available"] is True
    assert answer["probability"] == 0.0


@pytest.mark.parametrize(
    "params, fragment",
    [
        ({"pcm": _b64(np.zeros(SR * 3, dtype="float32"))}, "too large"),  # > 2 s soft cap
        ({"pcm": "AAAA", "sample_rate": 44100}, "16 kHz"),
        ({"pcm": "AAAA", "threshold": "loud"}, "threshold"),
    ],
)
def test_bad_input_is_a_4001(params, fragment):
    answer = _call(**params)

    assert answer["error"]["code"] == 4001
    assert fragment in answer["error"]["message"]


def test_missing_vad_fails_open_instead_of_erroring(monkeypatch):
    """Lean installs (no faster-whisper) must answer available:false, not raise."""
    monkeypatch.setattr(speech_gate, "available", lambda: False)

    assert _call(pcm=_b64(_clatter()), sample_rate=SR)["result"] == {"available": False}
