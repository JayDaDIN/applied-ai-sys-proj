# Playlist-Aware Music Recommender with RAG

## Original Project (Modules 1–3)

This project extends my **Music Recommender Simulation** from Modules 1–3. The original was a CLI-first classroom simulation that recommended fictional songs from a 30-song CSV catalog by comparing each song's genre, mood and audio features (energy, tempo, valence, danceability, acousticness) with a hand-written listener profile. It used a transparent weighted scoring rule to rank songs and print the reasons for each match. The goal was to make a recommender's logic easy to inspect, not to predict real listening behavior.

## Title and Summary

**Playlist-Aware Music Recommender with RAG** adds retrieval-augmented generation (RAG) to the original recommender. Instead of typing preferences by hand, a listener supplies their **actual playlist** (a CSV or a Spotify export). The system:

1. builds a taste profile from the playlist,
2. retrieves the best-matching candidates from the catalog, and optionally background on the listener's artists from the **web**, then
3. has **Claude** pick and explain the final recommendations using only those retrieved songs.

This matters because a hand-written profile is a guess about taste, while a playlist is evidence of it. Grounding the language model in retrieved catalog songs keeps its suggestions real and checkable: it can explain and reorder songs, but it cannot invent tracks that don't exist.

## Architecture Overview

The full diagram is in [`diagrams/architecture.mmd`](diagrams/architecture.mmd) (Mermaid source). It has five groups of components:

- **Input:** the listener's playlist CSV, the song catalog (`data/songs.csv`) and the CLI flags.
- **Retriever:** `src/playlist.py` loads the playlist and averages its audio features into a taste profile. Fine-grained Spotify genres such as "lofi beats" are mapped to the catalog's labels, such as `lofi`. The original rule-based scorer (`score_song`) then ranks the catalog and returns the top-N candidates, skipping songs the listener already has. With `--web`, Claude's web search tool also gathers short notes on the playlist's top artists and genres.
- **Generator / Agent:** `src/rag.py` builds a prompt from the playlist summary, the taste profile, the candidates and the web notes. Claude (`claude-opus-5-5`) returns a ranked pick list as schema-constrained JSON, with a reason for each pick.
- **Evaluator / Guardrails:** these steps check Claude's output before anything is shown.
  - The refusal check catches a declined request; a server-side fallback model is also enabled.
  - `validate_picks` keeps only song IDs that were actually retrieved, drops duplicates, and logs a warning for every dropped pick.
  - If Claude is unreachable, refuses, returns invalid JSON or returns no valid picks, or if `--no-llm` is set, the system falls back to the rule-based ranking and says so.
- **Human and testing checks:** the pytest suite checks the loader, the scorer, the validator and the web-retrieval parsing with a fake Claude client. A human reviews the printed reasons and web sources against the rule-based scores, and adjusts the weights, the playlist or the prompt when the picks look off. The log file (`logs/recommender.log`) records token usage, dropped picks and fallbacks for that review.

Data flows **playlist → taste profile → retrieved candidates (+ web notes) → Claude → validation → CLI output → human review**.

### How the original scoring works

Each preference produces a match from 0 to 1. Genre and mood are compared case-insensitively after trimming whitespace. Numeric features use `max(0, 1 - |song - target|)`, and tempo is scaled by 80 BPM.

| Feature | Weight |
|---|---:|
| Genre | 20 |
| Mood | 20 |
| Energy, Tempo, Valence, Danceability, Acousticness | 12 each |

The weighted matches are divided by the total weight of the preferences supplied and scaled to 0–100, so a profile with missing preferences still scores on the same scale. Ties are broken by song ID. In RAG mode, this score is the **retriever**, and it is also shown next to each of Claude's picks so the two can be compared.

## Setup Instructions

Run every command from the repository root.

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Windows PowerShell
   source .venv/bin/activate       # macOS/Linux
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the original seven sample profiles** (no API key needed):

   ```bash
   python -m src.main
   ```

4. **Run playlist retrieval without the LLM** (no API key needed):

   ```bash
   python -m src.main --playlist data/sample_playlist.csv --no-llm
   ```

5. **Set an Anthropic API key** to enable Claude:

   ```bash
   export ANTHROPIC_API_KEY=...            # PowerShell: $env:ANTHROPIC_API_KEY="..."
   ```

6. **Run full RAG:**

   ```bash
   python -m src.main --playlist data/sample_playlist.csv          # playlist retrieval + Claude curation
   python -m src.main --playlist data/sample_playlist.csv --web    # also retrieve web context
   ```

   Options:

   | Flag | Effect |
   |---|---|
   | `-k N` | Number of recommendations (default 5) |
   | `--candidates N` | Catalog songs Claude chooses from (default 10) |
   | `--catalog PATH` | Use a different song catalog (default `data/songs.csv`) |
   | `-v` / `--verbose` | Show INFO log messages on the console |
   | `--debug` | Show DEBUG messages, including the full prompt sent to Claude |
   | `--log-file PATH` | Log file location (default `logs/recommender.log`); `none` turns it off |

7. **Run the tests:**

   ```bash
   python -m pytest -q
   ```

**Using your own playlist:** export it to CSV with a tool such as [Exportify](https://exportify.app/), or write a CSV with the catalog's columns. Column names are matched case-insensitively, and only a title column is required.

## Sample Interactions

Examples 1–3 are real outputs from the current code. Example 4 is the Claude-curated output and must come from a run with an API key.

### 1. Spotify-style lofi playlist → retrieval only

**Input:** `python -m src.main --playlist data/sample_playlist.csv --no-llm -k 3`, using a 6-track export with genres like "lofi beats", "chillhop" and "indie folk".

```text
Derived taste profile: {'genre': 'lofi', 'target_energy': 0.337, 'target_tempo_bpm': 76.167, 'target_valence': 0.588, 'target_danceability': 0.563, 'target_acousticness': 0.835}
Top artists: Kettle & Clay, Paper Lanterns, Moss Radio, Lumen Field, Harbor Hum

Recommendations (rule-based only, from 10 retrieved candidates)

1. Focus Flow - LoRoom
   Genre: lofi | Mood: focused
   Rule-based match score: 96.9/100
2. Midnight Coding - LoRoom
   Genre: lofi | Mood: chill
   Rule-based match score: 95.3/100
3. Corner Table - Blue Finch
   Genre: jazz | Mood: chill
   Rule-based match score: 71.3/100
```

The Spotify genres were mapped to the catalog label `lofi`. *Library Rain* is the catalog's best lofi match, but it is already on the playlist, so it was correctly left out.

### 2. Catalog-format upbeat pop playlist → retrieval only

**Input:** `python -m src.main --playlist data/sample_playlist_upbeat.csv --no-llm -k 3`

```text
Derived taste profile: {'genre': 'pop', 'mood': 'happy', 'target_energy': 0.845, 'target_tempo_bpm': 122.5, 'target_valence': 0.835, 'target_danceability': 0.86, 'target_acousticness': 0.115}

1. Daybreak Parade - Color Theory     pop | happy        96.6/100
2. Electric Bloom - Chrome Garden     electronic | happy 77.0/100
3. Gym Hero - Max Pulse               pop | intense      75.8/100
```

The playlist includes *Sunrise City*, so it is excluded even though it would otherwise rank second.

### 3. Original hand-written profile with labels the catalog doesn't have

**Input:** `python -m src.main`, "Unseen Genre and Mood" profile (`"dream pop"`, `"nostalgic"`)

```text
1. After the Rain - Glass Orchard       indie pop | chill   56.8/100
2. Midnight Postcard - The Still Frames rock | relaxed      56.7/100
```

No categorical points are awarded, so numeric features alone drive the ranking. This is the gap that Claude's curation and the web notes are meant to fill.

### 4. Claude-curated recommendations with web retrieval

**Input:** `python -m src.main --playlist data/sample_playlist.csv --web -k 3`

```text
TODO: paste the real output here after running with ANTHROPIC_API_KEY set.
Expected sections: Claude's read on this listener, Web notes,
3 recommendations with "Why:" reasons, and Web sources.
```

## Logging and Error Handling

**Logging** uses Python's standard `logging` module. It is configured once, in `src/logging_config.py`, and every module logs through `logging.getLogger(__name__)`.

- **Console (stderr):** only warnings and errors by default, so the recommendations on stdout stay readable. `-v` adds INFO messages, and `--debug` adds DEBUG messages such as the full prompt. The console never shows tracebacks.
- **Log file:** `logs/recommender.log` records every run at INFO level (DEBUG with `--debug`), including tracebacks for unexpected errors. It rotates at about 1 MB and keeps 3 old files. The `logs/` folder is git-ignored.
- **What gets logged:**
  - run start and finish,
  - catalog and playlist sizes, and the derived taste profile,
  - every retrieved candidate with its score,
  - each Claude call's stop reason, token usage and request ID,
  - the number of web sources found,
  - every pick dropped by validation,
  - every fallback.

Example log lines:

```text
2026-11-18 20:46:54,250 INFO     src.rag: Taste profile: {'genre': 'lofi', 'target_energy': 0.337, ...}
2026-11-18 20:46:54,252 INFO     src.rag: Retrieved 10 candidates: 9:96.9, 2:95.3, 27:71.3, 20:71.1, ...
2026-11-18 20:47:02,420 WARNING  src.rag: Claude curation failed, showing rule-based results: No Anthropic credentials found. ...
```

**Error handling** is built on a small set of exception classes in `src/errors.py`:

| Error | Raised when | What the user sees |
|---|---|---|
| `DataLoadError` | A catalog or playlist file is missing, isn't UTF-8 or valid CSV, lacks a required column, has a non-numeric catalog value (the message gives the line number) or has no rows | `ERROR: <message>` and exit code 1 |
| `LLMError` | Claude can't be reached or the request fails: missing or invalid API key, rate limit, timeout, connection or server error, refusal, cut-off answer, invalid JSON | A warning, then rule-based results (the run still succeeds) |
| Bad CLI argument | For example `-k 0` | argparse usage message and exit code 2 |
| Anything else | A bug | `ERROR: Unexpected error; this is a bug.` on the console, the traceback in the log file, and exit code 3 |

Some problems are not fatal. A non-numeric cell in a playlist is logged and ignored. A failed web search lets the run continue without web notes. If Claude fails or returns no usable picks, the rule-based ranking is shown and the output says so.

## Design Decisions

- **The rule-based scorer is kept as the retriever.** Claude only chooses among candidates that the transparent scorer has already found. This keeps every recommendation grounded in the catalog, keeps the original project's explainability, and lets each AI pick be compared with a rule-based score. *Trade-off:* Claude can't surface a good song that the scorer ranked outside the top N.
- **The output is validated, not just requested in the prompt.** The prompt tells Claude to use only candidate IDs, but `validate_picks` enforces it in code. Schema-constrained JSON makes the output machine-checkable. *Trade-off:* the structured-output mode can't also return inline citations, so web sources are collected from the search results and printed separately.
- **Web retrieval is a separate, optional step.** Web search runs before curation and produces plain notes that are passed in as untrusted data. This keeps the curation call deterministic in shape and lets `--web` be turned off. *Trade-off:* it costs an extra API call, and the web notes vary between runs.
- **Playlists come from CSV files, not a live Spotify connection.** A live connection needs app registration and an OAuth sign-in, and Spotify has restricted audio-feature data for new apps. A CSV export avoids both and works offline. *Trade-off:* the playlist is a snapshot, and some exports lack audio features, which gives a thinner profile.
- **Specific genres win when mapping Spotify genres to catalog labels.** "indie pop" maps to the catalog's `indie pop`, not `pop`, and only whole words count, so "rock" doesn't match "rockabilly".
- **The system degrades gracefully.** `--no-llm`, an empty pick list, a missing API key or a failed API call still leave a working rule-based recommender, with a visible warning explaining why. *Trade-off:* a run can succeed (exit code 0) without Claude, so check the warning or the log to confirm Claude was used.
- **Expected errors and bugs are handled differently.** Bad input files and API problems get a one-line message the user can act on. Unexpected exceptions get a full traceback in the log file, so the console stays readable without losing debugging detail.
- **Claude Opus 5.5 runs with low effort for web notes and medium effort for curation.** Note-taking is simple, and ranking needs more judgment. Server-side refusal fallback is enabled.

## Testing Summary

**What worked**

- All **41 tests pass** (`python -m pytest -q`):
  - 8 original tests for scoring and ranking,
  - 13 in `tests/test_rag.py` for the RAG pipeline,
  - 20 in `tests/test_errors_and_logging.py` for error handling and logging.
- The error-handling tests cover:
  - missing and malformed catalog and playlist files,
  - translation of SDK errors (missing key, rate limit, connection failure),
  - every fallback path (API failure, invalid JSON, cut-off answer, only invented song IDs, failed web search),
  - warnings logged for dropped picks,
  - CLI exit codes 0, 1, 2 and 3, including that the traceback goes to the log file and not the console.
- The RAG tests cover:
  - both playlist formats, missing columns and missing features,
  - genre mapping, including the specific-label rule,
  - exclusion of songs already on the playlist,
  - rejection of hallucinated and duplicate song IDs,
  - prompt contents,
  - `pause_turn` continuation and error results during web search,
  - refusals.
- The tests use a **fake Claude client**, so they run without an API key, network access or cost, and give the same result every run.
- I checked the request arguments against the installed `anthropic` SDK (2.10.0). A request with a deliberately invalid key reached the API and returned 401, so the client code builds and sends requests. Authentication is checked before the request body, so this doesn't show that the API accepts the request itself.

**What didn't work, or isn't verified yet**

- The first version of the error handling reported each failure two or three times on the console (logged as an error, logged again as a warning, then printed). I changed it so each problem appears once on the console, with the details in the log file.
- Two of my first test expectations were wrong: I guessed that *Midnight Coding* would rank first, but the scorer correctly ranks *Focus Flow* higher. I fixed the tests rather than the code, since the scorer was right.
- The **live Claude path has not been run end to end** with a real API key, so the quality of Claude's picks and reasons hasn't been evaluated yet (see Sample Interaction 4).
- The Mermaid diagram was reviewed by hand but not rendered by a tool.

**What I learned**

Testing an LLM feature works best when the AI call sits behind something you can swap out. The deterministic parts (parsing, retrieval, validation) get exact tests. The AI part gets contract tests: what goes in, and what happens with good, bad or refused output. The quality of Claude's choices still needs human review, because a unit test can't judge taste.

## Reflection

This project showed me that a lot of the work in adding AI to a system goes into the code around the model: deciding what it is allowed to see, constraining what it can return, and checking that output before trusting it. Keeping the original rule-based recommender as the retriever turned out to be the key decision. It gave the AI a grounded set of choices and gave me a baseline to compare its answers against. Breaking the problem into retrieve, generate and validate steps made each piece testable on its own. (My responsible-AI reflection is in [`model_card.md`](model_card.md).)

## Limitations and Risks

- The catalog has only 30 fictional tracks, so Claude can reorder and explain songs but can only choose from these.
- Genre and mood labels are subjective, and exact matching misses related labels such as "relaxed" and "chill".
- Audio features are rough stand-ins for how music sounds. The system doesn't analyze audio, lyrics or cultural context.
- The weights and the tempo scale are hand-picked assumptions, not learned from listener behavior.
- Web notes come from live search results. They can be wrong, out of date or biased toward well-known artists, and they change between runs.
- Each Claude-backed run makes one or two paid API calls.
- The starter dataclass `Recommender` methods are placeholders. The working path is the functional pipeline.

## Project Files

- `src/main.py`: CLI runner, sample profiles and playlist mode
- `src/recommender.py`: scoring and ranking logic, the retriever's scorer
- `src/playlist.py`: playlist CSV loader and taste-profile builder
- `src/rag.py`: RAG pipeline (playlist and web retrieval, Claude curation, validation, fallbacks)
- `src/errors.py`: exception classes (`RecommenderError`, `DataLoadError`, `LLMError`)
- `src/logging_config.py`: console and rotating log-file setup
- `tests/test_errors_and_logging.py`: error-handling, fallback, logging and exit-code tests
- `data/songs.csv`: fictional song catalog
- `data/sample_playlist.csv`: example Spotify-style playlist export
- `data/sample_playlist_upbeat.csv`: example catalog-format playlist
- `tests/test_recommender.py`: original scoring tests
- `tests/test_rag.py`: playlist and RAG pipeline tests (fake Claude client)
- `diagrams/architecture.mmd`: Mermaid system diagram
- `model_card.md`: intended use, data, evaluation, limitations and responsible-AI reflection
