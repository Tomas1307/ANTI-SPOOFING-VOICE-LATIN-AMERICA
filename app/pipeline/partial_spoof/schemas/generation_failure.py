"""
Schema for a recoverable generation failure recorded in the checkpoint.

Generation failures (CUDA OOM, model exception, NaN audio, zero-byte
output) are distinct from quality failures and are eligible for retry
with a bumped seed up to MAX_GENERATION_RETRIES.
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class GenerationFailure(BaseModel):
    """Record of a recoverable TTS generation failure for one sample.

    Stored inside CheckpointState.failed_generation, keyed by sample_key.
    Distinguishes genuine generation errors (worth retrying with a
    different seed) from quality failures (which are kept in the corpus
    under the keep-bad-stuff principle and never retried at this layer).

    Attributes:
        error: Short error description, typically the exception class
            name plus a truncated message head.
        retries: Number of retry attempts so far. Caps at
            MAX_GENERATION_RETRIES from settings.
        last_attempt_at: ISO timestamp of the most recent attempt
            (UTC).
    """

    error: str = Field(
        ...,
        description="Truncated exception class + message",
    )
    retries: int = Field(
        default=0,
        description="Retry attempt counter (capped at MAX_GENERATION_RETRIES)",
    )
    last_attempt_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the most recent attempt",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
