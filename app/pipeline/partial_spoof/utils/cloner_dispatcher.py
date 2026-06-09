"""
Lazy-import dispatcher for the per-attack Cloner classes.

Replaces the legacy ``strategy_factory`` that constructed duplicate
partial_spoof strategy wrappers. Now every attack has exactly one
Cloner -- defined inside ``<attack>_attack/utils/cloner.py`` and
subclassing ``app.utils.base_cloner.BaseCloner`` -- and both the
standalone Step 3 and partial_spoof Step 2 import from there.

Each branch imports lazily so the SDK heavy-loads (chatterbox, qwen_tts,
omnivoice, outetts, melo, openvoice, requests) only get pulled in for
the attack actually being run in the current process. This preserves
venv isolation: partial_spoof Step 2 runs inside the attack's own venv
(via the parallel launcher's subprocess dispatch), so the lazy import
only succeeds for the right attack -- exactly as the legacy strategy
factory behaved.
"""
from typing import Type

from app.utils.base_cloner import BaseCloner


def get_cloner_class(attack_system: str) -> Type[BaseCloner]:
    """Resolve the Cloner class for the requested attack.

    Imports are performed inside each branch so that loading this
    dispatcher does not pull in every TTS SDK at once. A given
    partial_spoof Step 2 run only imports the SDK matching its
    ATTACK_SYSTEM, which is also the only SDK installed inside the
    active venv.

    Args:
        attack_system: Lowercase attack identifier. Must match one of
            'chatterbox', 'qwen', 'fishgram', 'openvoice', 'outetts',
            'omnivoice'.

    Returns:
        The Cloner subclass (not an instance) for the requested attack.
        All Cloners subclass ``BaseCloner``; callers instantiate via
        ``CloneClass()`` and call ``load()``, ``prepare_speaker()``,
        ``clone_single()``, ``cleanup()`` per the documented contract.

    Raises:
        ValueError: If attack_system is not recognised.
        ImportError: If the attack's Cloner module cannot be imported
            in the current venv -- usually because the SDK is missing.
            Surfaced as a clear failure so the caller knows to switch
            venvs.
    """
    if attack_system == "chatterbox":
        from app.pipeline.chatterbox_attack.utils.cloner import Cloner
        return Cloner
    if attack_system == "qwen":
        from app.pipeline.qwen_attack.utils.cloner import Cloner
        return Cloner
    if attack_system == "fishgram":
        from app.pipeline.fishgram_attack.utils.cloner import Cloner
        return Cloner
    if attack_system == "openvoice":
        from app.pipeline.openvoice_attack.utils.cloner import Cloner
        return Cloner
    if attack_system == "outetts":
        from app.pipeline.outetts_attack.utils.cloner import Cloner
        return Cloner
    if attack_system == "omnivoice":
        from app.pipeline.omnivoice_attack.utils.cloner import Cloner
        return Cloner
    raise ValueError(
        f"Unknown attack_system: {attack_system!r}. Expected one of "
        "'chatterbox', 'qwen', 'fishgram', 'openvoice', 'outetts', 'omnivoice'."
    )
