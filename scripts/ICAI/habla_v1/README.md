# ICAI 2026 — HABLA v1 verification scripts

Scripts supporting the camera-ready revision of *Accent-Based Evaluation of
Speech Anti-spoofing Countermeasures Across Multiple Languages* (ICAI 2026,
Springer CCIS, paper 16).

These run on **ml-server03**, against the corpora under
`/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/03-asvspoof-mega/`.
They are read-only with respect to that directory.

## check_speaker_leakage.py

Verifies that no speaker appears in more than one partition, for the English,
Spanish and combined corpora.

The paper states that HABLA speakers are assigned exclusively to a single
partition. That statement is the defence against the overfitting concern raised
in review, so it needs to be demonstrated rather than asserted.

Speaker identity is recovered from the utterance identifier for Spanish
(`vem_03397` decodes as Venezuela / male / speaker 03397) and from
`protocol.txt` for English. Voice-conversion utterances name both a source and a
target speaker; both are counted, which is the strict criterion.

```bash
python3 scripts/ICAI/habla_v1/check_speaker_leakage.py
```

Exits 0 when clean, 1 when any overlap is found.

### Reading the output

- Overlap between **train and test** is serious and would require qualifying the
  reported Spanish results.
- Overlap between **train and val** is milder, since validation only drives
  early stopping, but still needs to be disclosed.
- Identical utterance identifiers across partitions would indicate a corpus
  construction error rather than a design decision.

One caveat on interpretation: if the partitions were defined on the target
speaker of each conversion only, source speakers may show up as overlaps. Check
whether reported leaks follow that pattern before treating them as real.
