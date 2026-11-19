"""Exception types for the recommender.

Every error the CLI expects to happen (bad input files, API problems) is a
``RecommenderError``, so ``main`` can catch them in one place and show a short
message instead of a traceback. Anything else is a bug and is logged with its
full traceback.
"""


class RecommenderError(Exception):
    """Base class for expected, user-facing errors."""


class DataLoadError(RecommenderError, ValueError):
    """A catalog or playlist file is missing, unreadable, or malformed.

    Also a ``ValueError`` so callers that catch bad-input errors still work.
    """


class LLMError(RecommenderError):
    """Claude could not be reached, declined, or returned unusable output."""
