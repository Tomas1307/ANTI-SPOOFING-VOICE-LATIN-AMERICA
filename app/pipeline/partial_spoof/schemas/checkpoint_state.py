"""
Persistent checkpoint state schema for resumable partial spoof runs.

Stored at OUTPUT_DIR/.checkpoint.json. Written atomically after every
successful WAV commit so a crash never loses more than the in-flight
sample. The CheckpointManager util handles atomic I/O and resume logic.
"""
from datetime import datetime, timezone
from typing import Dict, Set

from pydantic import BaseModel, Field

from app.pipeline.partial_spoof.schemas.generation_failure import GenerationFailure


class CheckpointState(BaseModel):
    """Persistent checkpoint state for one (attack, partition) run.

    Stored at OUTPUT_DIR/.checkpoint.json. Updated atomically via
    temp-file-and-rename after every successful WAV commit (clone in
    Step 2, spliced output in Step 5, jittered output in Step 5b) so a
    crash never loses more than the in-flight sample.

    Resume protocol:
      Step 2 skips sample_keys already present in `cloned`.
      Step 5 skips sample_keys already present in `spliced`.
      Step 5b skips sample_keys already present in `jittered`.
      Entries in `failed_generation` are retried until their counter
      reaches MAX_GENERATION_RETRIES from settings, after which they
      are abandoned (not raised, just left out of the corpus).

    Attributes:
        attack: Attack system this checkpoint belongs to. One of
            'fishgram', 'qwen', 'omnivoice', 'openvoice', 'chatterbox',
            'outetts'.
        partition: Partition this checkpoint belongs to. One of
            'not_jittered', 'jittered'.
        cloned: Sample keys whose Step 2 clone WAV is committed to disk.
        spliced: Sample keys whose Step 5 spliced WAV is committed to disk.
            A sample_key may appear with W1/W2/W3 suffixes; each tier
            output is checkpointed independently.
        jittered: Sample keys whose Step 5b jittered WAV is committed to
            disk (only populated when partition == 'jittered').
        failed_generation: Sample keys that raised a recoverable error
            in Step 2, with per-key retry state.
        last_updated: UTC timestamp of the last successful write.
    """

    attack: str = Field(
        ...,
        description="Attack system identifier",
    )
    partition: str = Field(
        ...,
        description="Corpus partition: 'not_jittered' or 'jittered'",
    )
    cloned: Set[str] = Field(
        default_factory=set,
        description="Sample keys with committed Step 2 clone WAVs",
    )
    spliced: Set[str] = Field(
        default_factory=set,
        description="Sample keys with committed Step 5 spliced WAVs",
    )
    jittered: Set[str] = Field(
        default_factory=set,
        description="Sample keys with committed Step 5b jittered WAVs",
    )
    failed_generation: Dict[str, GenerationFailure] = Field(
        default_factory=dict,
        description="Sample keys with recoverable Step 2 errors plus retry state",
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the most recent checkpoint write",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
