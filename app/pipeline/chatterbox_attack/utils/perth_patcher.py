"""
Module-level patch for the resemble-perth watermarker.

Importing this module replaces ``perth.PerthImplicitWatermarker`` with
:class:`~app.pipeline.chatterbox_attack.utils.watermark_remover.NoOpWatermarker`
so that Chatterbox can be instantiated without the native Perth binary.

The patch runs exactly once at import time. Subsequent imports are no-ops
thanks to Python's module caching.

Why this is necessary:
    ``ChatterboxMultilingualTTS.__init__`` calls
    ``perth.PerthImplicitWatermarker()`` unconditionally. On many systems the
    Perth native binary fails to load, leaving the callable as ``None`` and
    causing a ``TypeError``. Patching before importing ``chatterbox.mtl_tts``
    prevents the crash entirely.
"""
import perth

from app.pipeline.chatterbox_attack.utils.watermark_remover import NoOpWatermarker

perth.PerthImplicitWatermarker = NoOpWatermarker

ensure_patched = True
