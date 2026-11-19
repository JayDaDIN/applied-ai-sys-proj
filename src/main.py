"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import argparse
import logging
import sys
from pathlib import Path

if __package__:
    from .errors import RecommenderError
    from .logging_config import DEFAULT_LOG_FILE, setup_logging
    from .playlist import load_playlist
    from .rag import run_rag
    from .recommender import load_songs, recommend_songs
else:
    from errors import RecommenderError
    from logging_config import DEFAULT_LOG_FILE, setup_logging
    from playlist import load_playlist
    from rag import run_rag
    from recommender import load_songs, recommend_songs

logger = logging.getLogger(__name__)


USER_PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "target_energy": 0.88,
        "target_tempo_bpm": 130,
        "target_valence": 0.88,
        "target_danceability": 0.90,
        "target_acousticness": 0.10,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "target_energy": 0.38,
        "target_tempo_bpm": 76,
        "target_valence": 0.60,
        "target_danceability": 0.58,
        "target_acousticness": 0.85,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "target_energy": 0.90,
        "target_tempo_bpm": 145,
        "target_valence": 0.45,
        "target_danceability": 0.65,
        "target_acousticness": 0.12,
    },
    "Acoustic Folk": {
        "genre": "folk",
        "mood": "relaxed",
        "target_energy": 0.45,
        "target_tempo_bpm": 92,
        "target_valence": 0.78,
        "target_danceability": 0.60,
        "target_acousticness": 0.88,
    },
    "Conflicting Preferences": {
        "genre": "rock",
        "mood": "chill",
        "target_energy": 0.95,
        "target_tempo_bpm": 60,
        "target_valence": 0.10,
        "target_danceability": 0.95,
        "target_acousticness": 0.95,
    },
    "Sparse Case-Insensitive Genre": {
        "genre": " POP ",
    },
    "Unseen Genre and Mood": {
        "genre": "dream pop",
        "mood": "nostalgic",
        "target_energy": 0.55,
        "target_tempo_bpm": 105,
        "target_valence": 0.65,
        "target_danceability": 0.60,
        "target_acousticness": 0.50,
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Music recommender simulation")
    parser.add_argument(
        "--playlist",
        help="CSV of a listener's playlist (catalog columns or a Spotify/Exportify export). "
        "Enables retrieval-augmented recommendations.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Also retrieve background on the playlist's artists with Claude's web search.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Claude; rank the retrieved candidates with the rule-based scorer only.",
    )
    parser.add_argument("-k", type=_positive_int, default=5, help="Number of recommendations.")
    parser.add_argument(
        "--candidates",
        type=_positive_int,
        default=10,
        help="Catalog songs retrieved for Claude to choose from.",
    )
    parser.add_argument("--catalog", default="data/songs.csv", help="Song catalog CSV.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Show INFO log messages on the console."
    )
    verbosity.add_argument(
        "--debug", action="store_true", help="Show DEBUG log messages, including prompts."
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help=f"Log file path (default: {DEFAULT_LOG_FILE}). Use 'none' to disable.",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def run_playlist_mode(args: argparse.Namespace, songs) -> None:
    tracks = load_playlist(args.playlist)
    use_llm = not args.no_llm
    if args.web and not use_llm:
        logger.warning("--web needs Claude, so it is ignored with --no-llm.")
    result = run_rag(
        tracks,
        songs,
        k=args.k,
        n_candidates=args.candidates,
        use_llm=use_llm,
        use_web=args.web and use_llm,
    )

    print(f"\nPlaylist: {args.playlist} ({result.playlist_summary['track_count']} tracks)")
    print("=" * 60)
    print(f"Derived taste profile: {result.profile}")
    if result.playlist_summary["top_artists"]:
        print(f"Top artists: {', '.join(result.playlist_summary['top_artists'])}")
    if result.listener_summary:
        print(f"\nClaude's read on this listener: {result.listener_summary}")
    if result.web and result.web.notes:
        print(f"\nWeb notes:\n{result.web.notes}")

    if result.used_llm:
        mode = "curated by Claude"
    elif result.warnings:
        mode = "rule-based only; Claude step failed, see warning"
    else:
        mode = "rule-based only"
    print(f"\nRecommendations ({mode}, from {len(result.candidates)} retrieved candidates)")
    for rank, pick in enumerate(result.picks, start=1):
        song = pick.song
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"   Rule-based match score: {pick.rule_score:.1f}/100")
        print(f"   Why: {pick.reason}")

    if result.web and result.web.sources:
        print("\nWeb sources:")
        for title, url in result.web.sources:
            print(f"  - {title}: {url}")

    # result.warnings were already logged to the console as WARNING lines.


def run_profile_mode(args: argparse.Namespace, songs) -> None:
    for profile_name, user_prefs in USER_PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=args.k)

        print(f"\n{profile_name}")
        print("=" * 60)
        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"\n{rank}. {song['title']} - {song['artist']}")
            print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
            print(f"   Match score: {score:.1f}/100")
            print(f"   Reasons: {explanation}")


def main(argv=None) -> int:
    """Run the CLI and return a process exit code.

    0 = success, 1 = expected error (bad input file, etc.),
    2 = bad command-line arguments (from argparse), 3 = unexpected bug,
    130 = interrupted with Ctrl+C.
    """
    args = parse_args(argv)
    if args.debug:
        console_level = file_level = logging.DEBUG
    else:
        console_level = logging.INFO if args.verbose else logging.WARNING
        file_level = logging.INFO
    log_file = None if args.log_file.lower() == "none" else Path(args.log_file)
    setup_logging(console_level, log_file, file_level)
    logger.info("Run started: %s", " ".join(sys.argv[1:] if argv is None else argv) or "(no args)")

    try:
        songs = load_songs(args.catalog)
        if args.playlist:
            run_playlist_mode(args, songs)
        else:
            run_profile_mode(args, songs)
    except RecommenderError as exc:
        logger.error("%s", exc)  # console shows "ERROR: <message>"
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception:
        # Console gets one line; the traceback goes to the log file only.
        where = f" Traceback written to {log_file}." if log_file else ""
        logger.exception("Unexpected error; this is a bug.%s", where)
        return 3

    logger.info("Run finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
