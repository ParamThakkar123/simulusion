from .dreamer_v1_reward import RewardModel
from .dreamer_v1_value import ValueModel

# Aliases for backward compatibility with lazy exports in simulusion/__init__.py
DreamerRewardModel = RewardModel
DreamerValueModel = ValueModel

__all__ = [
    "DreamerRewardModel",
    "DreamerValueModel",
    "RewardModel",
    "ValueModel",
]
