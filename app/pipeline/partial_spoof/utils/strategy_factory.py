"""
Factory function for creating attack strategy instances.

Uses conditional module imports to avoid loading all 6 TTS frameworks
at once. Each strategy module has its heavy ML imports at file top
(per CLAUDE.md). This factory only imports the selected strategy module.
"""
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy, VALID_ATTACK_SYSTEMS


def create_attack_strategy(attack_system: str) -> AttackStrategy:
    """Create an attack strategy instance for the specified voice cloning system.

    Imports only the selected strategy module to avoid loading unnecessary
    ML dependencies. Each concrete strategy file contains its own heavy
    imports (torch, model-specific libraries) at the top of its file.

    Args:
        attack_system: Voice cloning system identifier. Must be one of:
            'fishgram', 'qwen', 'cosyvoice', 'outetts', 'chatterbox', 'openvoice'.

    Returns:
        Concrete AttackStrategy instance for the specified system.

    Raises:
        ValueError: If attack_system is not a recognized system name.
    """
    if attack_system not in VALID_ATTACK_SYSTEMS:
        raise ValueError(
            f"Unknown attack system '{attack_system}'. "
            f"Valid options: {VALID_ATTACK_SYSTEMS}"
        )

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
