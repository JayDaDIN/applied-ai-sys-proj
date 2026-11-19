import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.playlist import build_taste_profile, load_playlist, summarize_playlist
from src.rag import (
    WebContext,
    build_curation_prompt,
    retrieve_candidates,
    retrieve_web_context,
    run_rag,
    validate_picks,
)
from src.errors import LLMError
from src.recommender import load_songs

ROOT = Path(__file__).resolve().parents[1]
SONGS = load_songs(str(ROOT / "data" / "songs.csv"))
SAMPLE_PLAYLIST = ROOT / "data" / "sample_playlist.csv"


class FakeMessages:
    """Stands in for client.beta.messages; returns queued responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def fake_client(*responses):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(responses)))


def text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_load_spotify_style_playlist():
    tracks = load_playlist(str(SAMPLE_PLAYLIST))
    assert len(tracks) == 6
    assert tracks[0]["title"] == "Library Rain"
    assert tracks[0]["artist"] == "Paper Lanterns"
    assert tracks[0]["tempo_bpm"] == 72.0


def test_load_catalog_style_playlist_and_missing_features(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("title,artist,genre,mood\nSong A,Someone,rock,intense\n", encoding="utf-8")
    tracks = load_playlist(str(path))
    assert tracks == [
        {
            "title": "Song A",
            "artist": "Someone",
            "genre": "rock",
            "mood": "intense",
            "energy": None,
            "tempo_bpm": None,
            "valence": None,
            "danceability": None,
            "acousticness": None,
        }
    ]
    profile = build_taste_profile(tracks, {"rock", "pop"})
    assert profile == {"genre": "rock", "mood": "intense"}


def test_playlist_without_title_column_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("artist\nSomeone\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_playlist(str(path))


def test_profile_maps_spotify_genres_to_catalog_labels():
    tracks = load_playlist(str(SAMPLE_PLAYLIST))
    profile = build_taste_profile(tracks, {s["genre"] for s in SONGS})
    assert profile["genre"] == "lofi"
    assert 0.3 < profile["target_energy"] < 0.4
    assert profile["target_acousticness"] > 0.8


def test_most_specific_genre_label_wins():
    tracks = [{"genre": "indie pop"}, {"genre": "indie pop"}, {"genre": "pop"}]
    profile = build_taste_profile(tracks, {"pop", "indie pop"})
    assert profile["genre"] == "indie pop"


def test_summary_counts_artists_and_genres():
    summary = summarize_playlist(load_playlist(str(SAMPLE_PLAYLIST)))
    assert summary["track_count"] == 6
    assert summary["top_artists"][0] == "Kettle & Clay"
    assert "lofi beats" in summary["top_genres"]


def test_retrieval_skips_songs_already_on_playlist():
    tracks = load_playlist(str(SAMPLE_PLAYLIST))
    profile = build_taste_profile(tracks, {s["genre"] for s in SONGS})
    candidates = retrieve_candidates(profile, SONGS, tracks, n=10)
    titles = [song["title"] for song, _, _ in candidates]
    assert "Library Rain" not in titles
    assert titles[0] == "Focus Flow"
    assert len(candidates) == 10


def test_validate_picks_drops_unknown_and_duplicate_ids():
    candidates = [(SONGS[0], 90.0, ""), (SONGS[1], 80.0, "")]
    data = {
        "picks": [
            {"song_id": 999, "reason": "hallucinated"},
            {"song_id": SONGS[1]["id"], "reason": "good"},
            {"song_id": SONGS[1]["id"], "reason": "dup"},
            {"song_id": SONGS[0]["id"], "reason": "also good"},
        ]
    }
    picks = validate_picks(data, candidates, k=5)
    assert [p.song["id"] for p in picks] == [SONGS[1]["id"], SONGS[0]["id"]]
    assert picks[0].reason == "good"


def test_prompt_includes_candidates_and_web_notes():
    candidates = [(SONGS[1], 88.0, "")]
    prompt = build_curation_prompt(
        {"track_count": 1, "top_artists": [], "top_genres": []},
        {"genre": "lofi"},
        candidates,
        WebContext(notes="Lofi is mellow."),
        k=1,
    )
    assert '"song_id": 2' in prompt
    assert "<web_notes>\nLofi is mellow.\n</web_notes>" in prompt


def test_run_rag_without_llm_is_rule_based():
    tracks = load_playlist(str(SAMPLE_PLAYLIST))
    result = run_rag(tracks, SONGS, k=3, use_llm=False)
    assert not result.used_llm
    assert len(result.picks) == 3
    assert result.picks[0].song["title"] == "Focus Flow"


def test_run_rag_with_llm_uses_claude_ordering():
    tracks = load_playlist(str(SAMPLE_PLAYLIST))
    answer = {
        "picks": [
            {"song_id": 27, "reason": "Jazzy, mellow, fits the jazz hop."},
            {"song_id": 2, "reason": "Classic lofi."},
        ],
        "listener_summary": "Likes calm study music.",
    }
    client = fake_client(text_response(json.dumps(answer)))
    result = run_rag(tracks, SONGS, k=2, use_llm=True, client=client)

    assert result.used_llm
    assert [p.song["title"] for p in result.picks] == ["Corner Table", "Midnight Coding"]
    assert result.listener_summary == "Likes calm study music."
    call = client.beta.messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in call


def test_web_retrieval_collects_notes_and_sources_across_pause_turn():
    search_result = SimpleNamespace(
        type="web_search_tool_result",
        content=[SimpleNamespace(title="Lofi guide", url="https://example.com/lofi")],
    )
    error_result = SimpleNamespace(type="web_search_tool_result", content=SimpleNamespace(error_code="max_uses_exceeded"))
    first = SimpleNamespace(stop_reason="pause_turn", content=[search_result])
    second = SimpleNamespace(
        stop_reason="end_turn",
        content=[error_result, SimpleNamespace(type="text", text="Mellow and jazzy.")],
    )
    client = fake_client(first, second)
    context = retrieve_web_context(
        client, {"top_artists": ["Kettle & Clay"], "top_genres": ["lofi beats"]}, {}
    )
    assert context.notes == "Mellow and jazzy."
    assert context.sources == [("Lofi guide", "https://example.com/lofi")]
    assert len(client.beta.messages.calls) == 2
    assert client.beta.messages.calls[0]["tools"][0]["type"] == "web_search_20260209"


def test_refusal_raises():
    client = fake_client(SimpleNamespace(stop_reason="refusal", content=[], stop_details=None))
    with pytest.raises(LLMError):
        retrieve_web_context(client, {"top_artists": [], "top_genres": []}, {})
