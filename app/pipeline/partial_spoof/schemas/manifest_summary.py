"""
Aggregate summary statistics for a generated partial spoof manifest.

Computed once at the end of manifest generation. Persisted next to the
manifest CSV (as partial_spoof_plan_summary.json) to provide an audit
trail for the paper: corpus marginals vs target weights, speaker
coverage, and tier yield potential.
"""
from typing import Dict

from pydantic import BaseModel, Field


class ManifestSummary(BaseModel):
    """Aggregate statistics over a partial spoof dispatch manifest.

    The summary lets us verify (before launching a multi-day run) that
    the probabilistic per-speaker assignment did not produce pathological
    skews. It also feeds the production-runs.md page so the paper can
    cite both target and realised marginals.

    Attributes:
        total_entries: Total rows in the manifest (one per bonafide file
            that survived the MIN_WORDS_W1 filter).
        per_attack_count: Number of files assigned to each attack.
        per_partition_count: Files per partition ('not_jittered' / 'jittered').
        per_attack_per_partition_count: Cross-tabulation of attack
            against partition counts; keyed by attack -> partition -> count.
        per_tier_potential_count: Total tier slots planned across the
            manifest (sum of len(planned_tiers) per row, bucketed per
            tier). This is the upper bound on output count if every
            generation and splice succeeds.
        speakers_total: Distinct speakers appearing in the manifest.
        speakers_with_all_attacks: Speakers whose files cover all six
            attacks at least once (large-file-count tail).
        speakers_with_single_attack: Speakers assigned to a single attack
            only (small-file-count tail; expected for speakers with very
            few utterances).
        attack_weights_target: Target distribution used during
            assignment, e.g. {'omnivoice': 0.40, 'qwen': 0.20, ...}.
        attack_weights_actual: Realised fraction per attack after the
            probabilistic draw, for direct comparison against the target.
    """

    total_entries: int = Field(
        ...,
        description="Total rows in the manifest after tier filtering",
    )
    per_attack_count: Dict[str, int] = Field(
        ...,
        description="Files assigned per attack",
    )
    per_partition_count: Dict[str, int] = Field(
        ...,
        description="Files per partition ('not_jittered' / 'jittered')",
    )
    per_attack_per_partition_count: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Attack x partition cross-tabulation",
    )
    per_tier_potential_count: Dict[str, int] = Field(
        ...,
        description="Planned tier slots per W1/W2/W3 (upper bound on outputs)",
    )
    speakers_total: int = Field(
        ...,
        description="Distinct speakers in the manifest",
    )
    speakers_with_all_attacks: int = Field(
        ...,
        description="Speakers covering all six attacks",
    )
    speakers_with_single_attack: int = Field(
        ...,
        description="Speakers assigned to only one attack",
    )
    attack_weights_target: Dict[str, float] = Field(
        ...,
        description="Target probability weights used during assignment",
    )
    attack_weights_actual: Dict[str, float] = Field(
        ...,
        description="Realised fraction per attack post-assignment",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
