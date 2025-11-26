# opponents/__init__.py

from .base import OpponentBase
from .simple_follow import SimpleFollowOpponent
from .predictive import PredictiveOpponent


def get_opponent(name: str) -> OpponentBase:
    """
    根據名稱回傳對手 AI 物件。

    可用名稱：
      - "simple-easy"    : 慢速、弱
      - "simple-normal"  : 一般速度
      - "simple-fast"    : 快速跟球
      - "predictive-normal" : 有預判、normal 強度
      - "predictive-hard"   : 有預判、較強
    """
    key = name.lower()

    if key == "simple-easy":
        return SimpleFollowOpponent(speed=3.0)
    if key == "simple-normal":
        return SimpleFollowOpponent(speed=5.0)
    if key == "simple-fast":
        return SimpleFollowOpponent(speed=7.0)

    if key == "predictive-normal":
        return PredictiveOpponent(
            speed=6.0,
            reaction_delay=3,
            anticipation=4,
        )
    if key == "predictive-hard":
        return PredictiveOpponent(
            speed=8.0,
            reaction_delay=1,
            anticipation=6,
        )

    # default：給一個中等強度
    return SimpleFollowOpponent(speed=5.0)
