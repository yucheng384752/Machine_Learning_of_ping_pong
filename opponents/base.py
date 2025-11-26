# opponents/base.py

from __future__ import annotations

from typing import Protocol


class OpponentBase(Protocol):
    """
    所有 2P 對手 AI 必須實作的介面。
    env 會把自己傳進來，讓對手讀取 ball_x, ball_y, p2_x 等資訊。
    """

    def reset(self, env) -> None:  # 可以選擇忽略，不需要一定有狀態
        ...

    def update(self, env) -> None:
        """
        根據目前環境狀態，更新 env.p2_x。
        不需要回傳任何東西，直接改 env 即可。
        """
        ...
