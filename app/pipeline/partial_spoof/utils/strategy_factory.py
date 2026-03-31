"""Factory for creating voice cloning attack strategy instances.

Uses conditional module imports to avoid loading all TTS frameworks
at once. Each strategy module imports its heavy ML dependencies at
file-level (per CLAUDE.md), so only the selected system's dependencies
are loaded into memory.
"""
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy


VALID_SYSTEMS = frozenset({
    "fishgram",
    "qwen",
    "cosyvoice",
    "outetts",
    "chatterbox",
    "openvoice",
})


def create_attack_strategy(attack_system: str) -> AttackStrategy:
    """Create and return the appropriate AttackStrategy for the given system.

    Args:
        attack_system: Voice cloning system identifier. Must be one of:
            fishgram, qwen, cosyvoice, outetts, chatterbox, openvoice.

    Returns:
        An instance of the corresponding AttackStrategy subclass.

    Raises:
        ValueError: If attack_system is not a recognized identifier.
    """
    if attack_system not in VALID_SYSTEMS:
        raise ValueError(
            f"Unknown attack system '{attack_system}'. "
            f"Valid options: {sorted(VALID_SYSTEMS)}"
        )

    # Conditional module imports to load only the selected strategy's
    # ML dependencies. Each strategy file has its heavy imports at
    # file-level per CLAUDE.md; here we import the module itself.
    if attack_system == "fishgram":
        from app.pipeline.partial_spoof.strategies.fishgram_strategy import FishGramStrategy
        return FishGramStrategy()

    if attack_system == "qwen":
        from app.pipeline.partial_spoof.strategies.qwen_strategy import QwenStrategy
        return QwenStrategy()

    if attack_system == "cosyvoice":
        from app.pipeline.partial_spoof.strategies.cosyvoice_strategy import CosyVoiceStrategy
        return CosyVoiceStrategy()

    if attack_system == "outetts":
        from app.pipeline.partial_spoof.strategies.outetts_strategy import OuteTTSStrategy
        return OuteTTSStrategy()

    if attack_system == "chatterbox":
        from app.pipeline.partial_spoof.strategies.chatterbox_strategy import ChatterboxStrategy
        return ChatterboxStrategy()

    if attack_system == "openvoice":
        from app.pipeline.partial_spoof.strategies.openvoice_strategy import OpenVoiceStrategy
        return OpenVoiceStrategy()

    raise ValueError(f"No strategy implementation for '{attack_system}'.")
