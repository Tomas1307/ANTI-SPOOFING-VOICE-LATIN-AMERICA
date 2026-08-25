"""
Detector training pipeline for the MARSA corpus.

One shared harness serves every detector: the corpus audit, protocol
resolution, training loop, checkpointing and evaluation are backend-agnostic.
Each detector contributes only a subpackage holding its model adapter and its
own settings.

This package deliberately re-exports nothing. Eager re-exports would make
importing the corpus auditor pull in torch, transformers and soundfile, and
the audit is designed to run on a machine with none of them. Import the
symbol you need from its own module.
"""
