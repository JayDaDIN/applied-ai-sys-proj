"""Load a listener's real playlist and turn it into a taste profile.

Two CSV layouts are accepted:

* The catalog layout used by ``data/songs.csv`` (``title``, ``artist``,
  ``genre``, ``mood``, ``energy``, ...).
* A Spotify-style export such as the one produced by Exportify
  (``Track Name``, ``Artist Name(s)``, ``Genres``, ``Energy``, ``Tempo``, ...).

Column names are matched case-insensitively, and every audio-feature column
is optional. A playlist with only titles and artists still loads; its profile
will just carry fewer preferences.
"""

import csv
import logging
from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List, Optional

if __package__:
    from .errors import DataLoadError
else:
    from errors import DataLoadError

logger = logging.getLogger(__name__)

# Canonical field -> accepted column names (compared after casefolding).
COLUMN_ALIASES = {
    "title": ("title", "track name", "track", "name", "song"),
    "artist": ("artist", "artist name(s)", "artist name", "artists"),
    "genre": ("genre", "genres", "artist genres"),
    "mood": ("mood",),
    "energy": ("energy",),
    "tempo_bpm": ("tempo_bpm", "tempo", "bpm"),
    "valence": ("valence",),
    "danceability": ("danceability",),
    "acousticness": ("acousticness",),
}

NUMERIC_FEATURES = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")


def _to_float(value: Optional[str], where: str = "") -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring non-numeric value %r%s", value, f" ({where})" if where else "")
        return None


def load_playlist(csv_path: str) -> List[Dict]:
    """Read a playlist CSV into normalized track dictionaries.

    Raises DataLoadError if the file can't be read, has no title column, or
    contains no tracks. Bad numeric cells are logged and skipped.
    """
    try:
        csv_file = open(csv_path, newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DataLoadError(f"Cannot open playlist {csv_path!r}: {exc.strerror or exc}") from exc

    try:
        with csv_file:
            tracks = _read_playlist_rows(csv_file, csv_path)
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"Playlist {csv_path!r} is not UTF-8 text: {exc}") from exc
    except csv.Error as exc:
        raise DataLoadError(f"Playlist {csv_path!r} is not valid CSV: {exc}") from exc

    if not tracks:
        raise DataLoadError(f"Playlist {csv_path!r} has no tracks.")
    logger.info("Loaded %d tracks from playlist %s", len(tracks), csv_path)
    return tracks


def _read_playlist_rows(csv_file, csv_path: str) -> List[Dict]:
    reader = csv.DictReader(csv_file)
    header_lookup = {
        (name or "").strip().casefold(): name for name in (reader.fieldnames or [])
    }
    column_for = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in header_lookup:
                column_for[field] = header_lookup[alias]
                break

    if "title" not in column_for:
        raise DataLoadError(
            f"Playlist {csv_path!r} needs a title column "
            "(for example 'title' or 'Track Name')."
        )
    missing_features = [f for f in NUMERIC_FEATURES if f not in column_for]
    if missing_features:
        logger.info("Playlist has no column for: %s", ", ".join(missing_features))

    tracks = []
    for line_number, row in enumerate(reader, start=2):  # line 1 is the header
        title = (row.get(column_for["title"]) or "").strip()
        if not title:
            logger.warning("Skipping playlist line %d: no title", line_number)
            continue
        track = {"title": title}
        for field in ("artist", "genre", "mood"):
            raw = row.get(column_for[field]) if field in column_for else None
            track[field] = (raw or "").strip()
        for field in NUMERIC_FEATURES:
            raw = row.get(column_for[field]) if field in column_for else None
            track[field] = _to_float(raw, where=f"line {line_number}, {field}")
        tracks.append(track)
    return tracks


def _split_multi(value: str) -> List[str]:
    """Split comma/semicolon separated genre or artist lists."""
    parts = value.replace(";", ",").split(",")
    return [part.strip().casefold() for part in parts if part.strip()]


def _pick_genre(tracks: Iterable[Dict], catalog_genres: Iterable[str]) -> Optional[str]:
    """Choose the playlist genre that best lines up with the catalog's labels.

    Spotify genres are fine-grained ("indie folk", "chill lofi beats"), while
    the catalog uses short labels ("folk", "lofi"). Each playlist genre votes
    for every catalog label it contains; if none line up, fall back to the
    most common raw playlist genre.
    """
    raw_counts = Counter()
    for track in tracks:
        raw_counts.update(_split_multi(track.get("genre", "")))
    if not raw_counts:
        return None

    catalog = {genre.strip().casefold() for genre in catalog_genres if genre.strip()}
    catalog_votes = Counter()
    for raw_genre, count in raw_counts.items():
        # Whole-word match; the longest label wins ("indie pop" over "pop").
        padded = f" {raw_genre} "
        matching = [label for label in catalog if f" {label} " in padded]
        if matching:
            catalog_votes[max(matching, key=len)] += count
    if catalog_votes:
        return max(catalog_votes.items(), key=lambda item: (item[1], item[0]))[0]
    return raw_counts.most_common(1)[0][0]


def build_taste_profile(
    tracks: List[Dict], catalog_genres: Iterable[str] = ()
) -> Dict:
    """Summarize a playlist as a preference dict that ``score_song`` accepts."""
    if not tracks:
        raise DataLoadError("Playlist is empty.")

    profile: Dict = {}
    genre = _pick_genre(tracks, catalog_genres)
    if genre:
        profile["genre"] = genre

    moods = Counter(t["mood"].casefold() for t in tracks if t.get("mood"))
    if moods:
        profile["mood"] = moods.most_common(1)[0][0]

    for feature in NUMERIC_FEATURES:
        values = [t[feature] for t in tracks if t.get(feature) is not None]
        if values:
            profile[f"target_{feature}"] = round(mean(values), 3)
    return profile


def summarize_playlist(tracks: List[Dict], max_items: int = 5) -> Dict:
    """Top artists and genres, used to steer web retrieval and the prompt."""
    artists = Counter()
    genres = Counter()
    for track in tracks:
        artists.update(
            name.strip() for name in track.get("artist", "").split(",") if name.strip()
        )
        genres.update(_split_multi(track.get("genre", "")))
    return {
        "track_count": len(tracks),
        "top_artists": [name for name, _ in artists.most_common(max_items)],
        "top_genres": [name for name, _ in genres.most_common(max_items)],
    }
