import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import main as cli
from src.errors import DataLoadError, LLMError, RecommenderError
from src.logging_config import setup_logging
from src.playlist import load_playlist
from src.rag import describe_api_error, run_rag, validate_picks
from src.recommender import load_songs

ROOT = Path(__file__).resolve().parents[1]
CATALOG = str(ROOT / "data" / "songs.csv")
PLAYLIST = str(ROOT / "data" / "sample_playlist.csv")
SONGS = load_songs(CATALOG)


class FailingMessages:
    """Fake client.beta.messages that raises, or returns queued responses."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    def create(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_with(*outcomes):
    return SimpleNamespace(beta=SimpleNamespace(messages=FailingMessages(*outcomes)))


def text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


NO_CREDENTIALS = TypeError("Could not resolve authentication method. Expected one of api_key ...")


# ---------- error hierarchy ----------

def test_data_load_error_is_a_value_error_and_recommender_error():
    assert issubclass(DataLoadError, ValueError)
    assert issubclass(DataLoadError, RecommenderError)
    assert issubclass(LLMError, RecommenderError)


# ---------- catalog loading ----------

def test_missing_catalog_file(tmp_path):
    with pytest.raises(DataLoadError, match="Cannot open song catalog"):
        load_songs(str(tmp_path / "nope.csv"))


def test_catalog_missing_column(tmp_path):
    path = tmp_path / "songs.csv"
    path.write_text("id,title\n1,A\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="missing column"):
        load_songs(str(path))


def test_catalog_bad_number_reports_line(tmp_path):
    header = "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
    path = tmp_path / "songs.csv"
    path.write_text(header + "1,A,B,pop,happy,0.5,120,0.5,0.5,0.5\n2,C,D,pop,happy,loud,120,0.5,0.5,0.5\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="line 3"):
        load_songs(str(path))


# ---------- playlist loading ----------

def test_missing_playlist_file(tmp_path):
    with pytest.raises(DataLoadError, match="Cannot open playlist"):
        load_playlist(str(tmp_path / "missing.csv"))


def test_empty_playlist(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("title,artist\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="no tracks"):
        load_playlist(str(path))


def test_bad_playlist_number_is_logged_and_skipped(tmp_path, caplog):
    path = tmp_path / "p.csv"
    path.write_text("title,energy\nSong,very high\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        tracks = load_playlist(str(path))
    assert tracks[0]["energy"] is None
    assert "very high" in caplog.text and "line 2" in caplog.text


# ---------- API error translation ----------

def test_missing_credentials_message():
    assert "ANTHROPIC_API_KEY" in describe_api_error(NO_CREDENTIALS)


def test_unrelated_errors_are_not_translated():
    assert describe_api_error(KeyError("x")) is None


def test_sdk_errors_are_translated():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert "connect" in describe_api_error(anthropic.APIConnectionError(request=request)).lower()
    response = httpx.Response(429, request=request)
    error = anthropic.RateLimitError("slow down", response=response, body=None)
    assert "429" in describe_api_error(error)


# ---------- graceful degradation in run_rag ----------

def test_curation_api_failure_falls_back_to_rule_based(caplog):
    tracks = load_playlist(PLAYLIST)
    with caplog.at_level(logging.WARNING):
        result = run_rag(tracks, SONGS, k=3, client=client_with(NO_CREDENTIALS))
    assert not result.used_llm
    assert [p.song["title"] for p in result.picks] == ["Focus Flow", "Midnight Coding", "Corner Table"]
    assert "ANTHROPIC_API_KEY" in result.warnings[0]
    assert "curation failed" in caplog.text


def test_invalid_json_falls_back():
    result = run_rag(load_playlist(PLAYLIST), SONGS, k=2, client=client_with(text_response("not json")))
    assert not result.used_llm
    assert "invalid JSON" in result.warnings[0]


def test_truncated_answer_falls_back():
    result = run_rag(
        load_playlist(PLAYLIST), SONGS, k=2,
        client=client_with(text_response('{"picks": [', stop_reason="max_tokens")),
    )
    assert not result.used_llm
    assert "cut off" in result.warnings[0]


def test_only_hallucinated_picks_falls_back():
    answer = json.dumps({"picks": [{"song_id": 999, "reason": "x"}], "listener_summary": "s"})
    result = run_rag(load_playlist(PLAYLIST), SONGS, k=2, client=client_with(text_response(answer)))
    assert not result.used_llm
    assert "no valid picks" in result.warnings[0]


def test_web_failure_continues_to_curation():
    answer = json.dumps({"picks": [{"song_id": 2, "reason": "lofi"}], "listener_summary": "s"})
    client = client_with(NO_CREDENTIALS, text_response(answer))
    result = run_rag(load_playlist(PLAYLIST), SONGS, k=1, use_web=True, client=client)
    assert result.used_llm
    assert result.web is None
    assert result.picks[0].song["id"] == 2
    assert "Web retrieval failed" in result.warnings[0]


def test_dropped_picks_are_logged(caplog):
    candidates = [(SONGS[0], 90.0, "")]
    with caplog.at_level(logging.WARNING):
        validate_picks({"picks": [{"song_id": 999, "reason": "x"}, "junk"]}, candidates, k=3)
    assert "not in retrieved candidates" in caplog.text
    assert "malformed pick" in caplog.text


# ---------- CLI exit codes and log file ----------

def test_cli_success_writes_log_file(tmp_path, capsys):
    log_file = tmp_path / "run.log"
    code = cli.main(["--catalog", CATALOG, "--playlist", PLAYLIST, "--no-llm", "-k", "2", "--log-file", str(log_file)])
    assert code == 0
    assert "Focus Flow" in capsys.readouterr().out
    log_text = log_file.read_text(encoding="utf-8")
    assert "Loaded 30 songs" in log_text and "Run finished successfully" in log_text
    logging.shutdown()


def test_cli_missing_playlist_returns_1(tmp_path, capsys):
    code = cli.main(["--catalog", CATALOG, "--playlist", str(tmp_path / "x.csv"), "--log-file", "none"])
    assert code == 1
    assert "ERROR: Cannot open playlist" in capsys.readouterr().err


def test_cli_unexpected_error_returns_3(monkeypatch, tmp_path, capsys):
    def boom(*args, **kwargs):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr(cli, "run_profile_mode", boom)
    log_file = tmp_path / "run.log"
    code = cli.main(["--catalog", CATALOG, "--log-file", str(log_file)])
    assert code == 3
    err = capsys.readouterr().err
    assert "this is a bug" in err
    assert "ZeroDivisionError" not in err  # console stays short; no traceback
    assert "ZeroDivisionError" in log_file.read_text(encoding="utf-8")  # traceback logged


def test_cli_rejects_non_positive_k():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["-k", "0", "--log-file", "none"])
    assert excinfo.value.code == 2


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    setup_logging(logging.WARNING, log_file=None)
