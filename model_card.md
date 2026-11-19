# Model Card: VibeFinder RAG (Playlist-Aware Music Recommender)

## 1. Model Name

**VibeFinder RAG**: a playlist-aware music recommender that uses retrieval-augmented generation. It extends my Modules 1–3 project, the **VibeFinder CLI Simulation**, a rule-based content recommender over a small fictional catalog.

## 2. Intended Use

VibeFinder RAG recommends songs from a 30-song fictional catalog based on a listener's real playlist. It is a classroom project for learning three things:

- how a transparent rule-based recommender works,
- how retrieval can ground a large language model in real data, and
- how to check AI output before trusting it.

It is **not** intended for real-world deployment, commercial recommendations, or claims about what a person will actually enjoy.

## 3. How the Model Works

The system works in three stages.

1. **Retrieve from the playlist.** The playlist CSV (in the catalog's own format or a Spotify/Exportify export) is turned into a taste profile.
   - Audio features (energy, tempo, valence, danceability, acousticness) are averaged across the playlist.
   - The most common genre is mapped onto the catalog's genre labels.
   - The original weighted scorer then ranks the catalog and returns the top 10 candidates, skipping songs already on the playlist.
2. **Retrieve from the web (optional).** Claude's web search tool writes short notes on the playlist's top artists and genres. The source URLs are kept.
3. **Generate and validate.**
   - Claude (`claude-opus-5-5`) receives the profile, the candidates and the web notes, and returns a ranked JSON pick list with a reason for each song.
   - Code then keeps only song IDs that were actually retrieved and drops duplicates.
   - If Claude fails, refuses, returns invalid JSON or returns no valid picks, the system falls back to the rule-based ranking and shows a warning.

**The original scorer:** genre and mood are worth 20 points each for an exact match. Energy, tempo, valence, danceability and acousticness are worth 12 points each, based on closeness to the target (tempo is scaled by 80 BPM). The total is normalized to 0–100 over the preferences that were actually provided.

## 4. Data

- **Catalog:** 30 fictional songs in `data/songs.csv`, with 12 genre labels and 6 mood labels. The feature values are illustrative, not measured from audio.
- **Playlists:** supplied by the user. Two samples are included: `data/sample_playlist.csv` (a Spotify-style lofi export) and `data/sample_playlist_upbeat.csv` (catalog-format pop).
- **Web notes:** live search results. They are not stored, and they can change between runs.
- **Data sent outside the machine:** when Claude is enabled, the playlist summary (top artists and genres), the taste profile and the candidate songs are sent to the Anthropic API. With `--web`, artist and genre names are also used as search queries. `--no-llm` keeps everything local.

## 5. Evaluation

- **41 automated tests pass:**
  - 8 original tests for scoring and ranking,
  - 13 for the RAG pipeline,
  - 20 for error handling and logging.
- The tests use a fake Claude client, so they check the pipeline's logic: parsing, retrieval, validation, fallbacks and exit codes. They do **not** measure the quality of Claude's recommendations.
- The retrieval-only mode was run on both sample playlists and on a deliberately mixed playlist (see Section 8).
- **Not yet evaluated:** the live Claude path has not been run end to end with a real API key, so the quality of the picks and explanations is still unmeasured.

## 6. Limitations and Biases

**Limitations**

- **Tiny, fictional catalog.** Claude can only reorder and explain 30 made-up songs. The playlist and web notes affect ranking and explanations, not what exists.
- **Averaging flattens taste.** The taste profile is the mean of the playlist's features. A playlist with two different moods gets a "middle" profile that describes neither (see Section 8).
- **Ties are broken arbitrarily.** When two genres appear equally often, the alphabetically later label wins, and mood ties go to whichever mood appears first. Half of a listener's taste can be silently ignored.
- **Exports often lack mood.** Spotify-style exports have no mood column, so the mood weight (20 of the 100 points) goes unused for those listeners.
- **Hand-picked weights.** The scoring weights and the 80 BPM tempo scale are my assumptions, not learned from listener behavior.
- **Unmeasured AI quality.** Nothing yet measures whether Claude's reordering is better than the rule-based ranking.

**Biases**

- **Catalog and label bias.** The catalog's genres are broad Western labels (pop, rock, lofi, folk, ...). Fine-grained or non-Western genres get mapped onto whatever label they contain, or onto none at all. Those listeners get worse matches.
- **Popularity bias from the web.** Search results are richer for well-known artists. Listeners of popular music get more detailed, more personal explanations, while listeners of niche or local artists get thin or generic notes.
- **Language-model bias.** Claude's reasons can reflect stereotypes about genres or the people who listen to them, and its wording can make a weak match sound convincing.
- **Designer bias.** I decided which features matter and how much: genre and mood together are 40% of the score. Someone who cares mostly about lyrics, language or vocals isn't represented at all.

## 7. Potential Misuse and Prevention

**How it could be misused**

- **Prompt injection.** Web pages, and even song titles in a playlist file, are text that reaches the model. A crafted page or title could try to make Claude ignore the listener or push a specific song.
- **Covert promotion.** Someone controlling the catalog, the weights or the prompt could quietly favor certain artists (a digital form of payola) while the explanations make the picks look personal.
- **Privacy exposure.** A playlist can reveal personal things: mood, religion, culture, life events. Running someone else's playlist without their consent sends that information to an outside API and to web search.
- **Cost abuse.** Each Claude run costs money. Running it in a loop, or on very large inputs, could run up an API bill.
- **Overtrust.** Users might treat confident AI explanations as facts about themselves.

**How I prevent or reduce it**

- **Claude can only choose from retrieved candidates.** This is enforced in code by `validate_picks`, not just requested in the prompt. Every dropped pick is logged as a warning, so injected or invented songs can't be shown and leave a trace.
- **Web notes are marked as untrusted.** The system prompt tells Claude to treat them as reference data, not instructions.
- **Rule-based scores are shown beside Claude's picks.** A strange reordering is visible to the user and easy to question.
- **Usage is limited and logged.** Web search is capped at 4 searches per run, each run makes at most a few calls, and the log records token usage per call. `--no-llm` runs fully offline.
- **Only summaries are sent.** Claude receives the top artists, top genres and the averaged profile, not the full playlist. The README states what leaves the machine.
- **Refusals are handled.** If Claude declines, the system falls back to rule-based results instead of retrying around the refusal.
- **If this were deployed:** I would add a consent notice before sending playlist data, rate limits per user, an audit of weight and catalog changes, and a label stating that the explanations are AI-generated.

## 8. What Surprised Me While Testing Reliability

- **A mixed playlist lost half its identity.** I tested a playlist with 2 heavy rock tracks and 2 ambient tracks. The top 5 contained **no ambient songs**. Both genres had 2 tracks each, and the tie went to "rock" only because of alphabetical order. Mood went to "intense" because it appeared first. The averaged energy (0.57) and acousticness (0.49) matched neither half. Nothing crashed and every test passed, yet the output was wrong for this listener. Passing tests didn't mean the system was reliable for every kind of listener.
- **The rule-based scorer beat my intuition.** I expected *Midnight Coding* to rank first for the lofi playlist, but *Focus Flow* scored higher (96.9 vs 95.3). Two of my tests failed because my expectation was wrong, not the code. The right fix was to change the tests, not the code.
- **"Missing API key" didn't look like an API error.** With no credentials, the Anthropic SDK raises a plain Python `TypeError` instead of one of its own error types. My first error handler would have reported it as a bug. I only found this by actually running the failure case.
- **A request "working" proved less than I thought.** A request with a deliberately invalid key reached the API and returned 401. That showed the client code runs, but authentication is checked before the request body, so it didn't prove the API would accept the request itself.
- **Fallbacks can hide failure.** Because the system falls back to rule-based results, a run where Claude failed still exits successfully. That's good for the user, but a run can look fine when the AI part didn't run at all. That's why the output says "rule-based only; Claude step failed, see warning", and the log records it.

## 9. Collaboration with AI

I built this project with **Claude Code**, an AI coding assistant, working in my repository. I described the goals: add RAG using a real playlist and the web, create a system diagram, write the README, and add logging and error handling. The assistant read my existing code, proposed a design, wrote the code and tests, and ran them. My job was to direct the work, review the changes, decide what fit the assignment, and check claims against real runs instead of trusting summaries.

**A suggestion I accepted:** keep my original rule-based scorer as the *retriever*, and only let Claude choose from the candidates it returns, with every returned song ID checked in code. I accepted this because it fit my project exactly. It reuses the transparent system I built in Modules 1–3, it makes hallucinated songs impossible to show instead of just unlikely, and it gives every AI pick a rule-based score to compare against. It also made the AI part testable without an API key.

**A suggestion I did not accept as written:** the first version of the error handling logged each Claude failure as an error, then logged it again as a warning, then printed it a third time in a "Notes" section. A missing file was also reported twice (`ERROR:` and `Error:`). It was technically thorough but harder to read, and a user seeing three copies of one message would think three things went wrong. The version I kept reports each problem **once** on the console, puts details and tracebacks only in the log file, and records the low-level API failure at INFO level, since the caller decides whether it's fatal.

**A suggestion that was out of scope:** connecting directly to Spotify with an OAuth sign-in. That would need app registration and a sign-in flow, and Spotify has restricted audio-feature data for new apps. A CSV export gave the same retrieval benefit with far less complexity, so the project uses CSV files.

**What I learned about working with AI:** the assistant was fast and thorough, but some of its first answers were wrong or over-built: two wrong test expectations, a placeholder `if True:` block that had to be rewritten, and the noisy error output. Running the code, reading the actual output and asking "is this what a user should see?" caught problems the AI's own summaries did not.

## 10. Future Work

- Handle mixed playlists by building several taste profiles (for example by clustering tracks) instead of one average, and break genre and mood ties by blending instead of alphabetical order.
- Run the live Claude path with an API key and compare Claude's picks with the rule-based ranking on a set of test playlists, ideally with feedback from real listeners.
- Add related-label matching (for example, "chill" ≈ "relaxed") and a larger, more diverse catalog.
- Add diversity controls, such as limiting repeated artists.
- Add a consent prompt before sending playlist data to external services.
