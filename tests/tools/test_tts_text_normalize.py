from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from tools.tts_text_normalize import prepare_spoken_text


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        raise AssertionError("not used")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def test_prepare_spoken_text_expands_celsius_and_weather_units():
    raw = """## Christchurch today\n\n- **Now:** about **14°C**, feels like **14°C**\n- **Wind:** 9 km/h\n- **Rain:** 1.3 mm\n- **Range:** 11\u201317°C\n"""

    spoken = prepare_spoken_text(raw)

    assert "##" not in spoken
    assert "**" not in spoken
    assert "14 degrees Celsius" in spoken
    assert "11 to 17 degrees Celsius" in spoken
    assert "9 kilometres per hour" in spoken
    assert "1.3 millimetres" in spoken
    assert "°C" not in spoken
    assert "km/h" not in spoken


def test_prepare_spoken_text_polish_edge_cases():
    # Heading folds into the next sentence as a lead-in, not a bare label.
    assert prepare_spoken_text("## Weather\nIt will be sunny") == "Weather, It will be sunny."
    # Bare degree unit (no leading number) still expands.
    assert "degrees Celsius" in prepare_spoken_text("measured in °C")
    # Trailing comma is not swallowed into the amount.
    assert "300 US dollars" in prepare_spoken_text("US$300, next")
    # Real numeric rates expand, but and/or, N/A, IDs and dates are left intact.
    assert "5 dollars per month" in prepare_spoken_text("$5/month")
    assert "and/or" in prepare_spoken_text("choose and/or option")
    assert "N/A" in prepare_spoken_text("status N/A here")
    assert "2026/06/02" in prepare_spoken_text("due 2026/06/02 ok")


def test_prepare_spoken_text_leaves_uppercase_m_counts_alone():
    # "82M" / "8M" are millions (parameters, users), not metres.
    assert prepare_spoken_text("Kokoro-82M is 327 MB") == "Kokoro-82M is 327 MB"
    assert prepare_spoken_text("a 70B model, 8M params, 5m away") == "a 70B model, 8M params, 5 metres away"
    assert prepare_spoken_text("100M users") == "100M users"
    assert prepare_spoken_text("5M vs 5m") == "5M vs 5 metres"
    assert prepare_spoken_text("1.5M rows, 5 M rows") == "1.5M rows, 5 M rows"
    assert prepare_spoken_text("5 MB over 2Mbps") == "5 MB over 2Mbps"


def test_prepare_spoken_text_still_expands_length_units():
    assert prepare_spoken_text("5m, 10 cm, 3mm and 90 km/h") == (
        "5 metres, 10 centimetres, 3 millimetres and 90 kilometres per hour")


def test_prepare_spoken_text_reads_tilde_as_about_only_before_a_number():
    assert prepare_spoken_text("roughly ~5 minutes") == "roughly about 5 minutes"
    assert prepare_spoken_text("wait ~ 5 minutes") == "wait about 5 minutes"
    assert prepare_spoken_text("~$5 each") == "about 5 dollars each"
    # An "about" already in the prose is not doubled.
    assert prepare_spoken_text("about ~5 minutes") == "about 5 minutes"
    assert prepare_spoken_text("~5km away").startswith("about 5")
    # Without a number, "~" is silent: never "about", never a spoken "tilde".
    assert prepare_spoken_text("~") == ""
    assert prepare_spoken_text("it's ~ fine") == "it's fine"
    assert prepare_spoken_text("~/notes is home, see ~/") == "is home, see"


def test_prepare_spoken_text_silences_paths_before_symbol_expansion():
    # Symbol/unit/rate expansion must not run inside a path: "~" became
    # "about", "2026/build" became "2026 per build", "5m/" became "5 metres/".
    assert prepare_spoken_text("config in ~/.hermes/profiles/jay/config.yaml today") == "config in today"
    assert prepare_spoken_text("run ./scripts/build.sh now") == "run now"
    assert prepare_spoken_text("run ./scripts/2026/build.sh now") == "run now"
    assert prepare_spoken_text("cd ~/v2/notes now") == "cd now"
    assert prepare_spoken_text("logs in /tmp/5m/run now") == "logs in now"
    # A spaced rate is not a path.
    assert prepare_spoken_text("$5 /month") == "5 dollars per month"


def test_prepare_spoken_text_strips_media_file_links():
    # "Open inference-server-shopping-list.xlsx" style tokens must never reach
    # the voice: hyphenated slugs + odd extensions make TTS loop ("eeeeee").
    raw = "The files are below.\nMEDIA:/Users/ricardo.mendes/Documents/inference-server-shopping-list.xlsx\nBye."
    spoken = prepare_spoken_text(raw)
    assert "MEDIA" not in spoken
    assert "shopping-list" not in spoken
    assert "xlsx" not in spoken
    assert "below" in spoken
    assert "Bye" in spoken


def test_prepare_spoken_text_keeps_sentence_break_after_inline_media_link():
    spoken = prepare_spoken_text("See MEDIA:/tmp/report-2026-q3.xlsx. Then reply.")
    assert "report" not in spoken
    assert spoken == "See. Then reply."


def test_prepare_spoken_text_closes_trailing_colons():
    # "the regex list:" + a now-removed raw token would leave the voice hanging
    # on an open colon-pause (the "aaaa" stutter). Close it with a period.
    spoken = prepare_spoken_text("Here is the list:\nMEDIA:/tmp/x.py\nMore text")
    assert "list:" not in spoken
    assert "list." in spoken


def test_prepare_spoken_text_closes_colon_on_single_line():
    # Multi-line text gets colons closed per line; single-line text reaches the
    # end-of-text rule instead.
    assert prepare_spoken_text("Here is the list:") == "Here is the list."
    # ...but a digit-preceded colon is a ratio and must stay intact.
    assert prepare_spoken_text("Final score 3:2") == "Final score 3:2"
