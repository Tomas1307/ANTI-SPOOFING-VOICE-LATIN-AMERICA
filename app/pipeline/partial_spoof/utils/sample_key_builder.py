"""
Build per-file sample_key strings for the partial spoof pipeline.

The sample_key is the primary key threaded through every step (Step 1
transcription, Step 2 cloning, Step 3 alignment, Step 4 selection,
Step 5 splicing, Step 5b jitter, Step 6 validation, Step 7 formatting)
and through the dispatch manifest. It must be unique per bonafide file
and stable across reruns so the checkpoint resume contract holds.

The legacy convention was `f"{speaker_id}_{audio_path.stem}"`, which
produced redundant double-prefix keys on HABLA v2 because v2 filenames
already include the speaker prefix (e.g. `arf_00295_00001008290.wav`).
This builder strips the redundancy when present while preserving the
legacy format for datasets whose filenames do NOT carry the speaker
prefix, so the pipeline stays portable across bonafide corpora.
"""


class SampleKeyBuilder:
    """Construct partial spoof sample_keys without double speaker prefix.

    Stateless helper. Single static method so both Step 1 (legacy mode)
    and the manifest pre-flight script can share one implementation and
    produce byte-identical keys.

    Example:
        >>> SampleKeyBuilder.build("arf_00295", "arf_00295_00001008290")
        'arf_00295_00001008290'
        >>> SampleKeyBuilder.build("arf_00295", "TEXT_00001")
        'arf_00295_TEXT_00001'
    """

    @staticmethod
    def build(speaker_id: str, audio_stem: str) -> str:
        """Return a sample_key, deduplicating an already-present speaker prefix.

        Args:
            speaker_id: HABLA speaker identifier (e.g. 'arf_00295').
            audio_stem: Audio filename without extension. May or may not
                begin with `speaker_id + "_"`.

        Returns:
            `audio_stem` unchanged if it already starts with
            `f"{speaker_id}_"`; otherwise `f"{speaker_id}_{audio_stem}"`
            (legacy behaviour for non-HABLA-v2 datasets).
        """
        prefix = f"{speaker_id}_"
        if audio_stem.startswith(prefix):
            return audio_stem
        return f"{prefix}{audio_stem}"
