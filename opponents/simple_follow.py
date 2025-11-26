# opponents/simple_follow.py

from __future__ import annotations

from .base import OpponentBase


class SimpleFollowOpponent:
    """
    最簡單版本：2P 永遠追著球的 x 位置跑。

    - 不做預判，也不區分球速
    - 只根據球中心與板子中心的相對位置，往左/往右移動
    """

    def __init__(self, speed: float = 5.0):
        self.speed = float(speed)

    def reset(self, env) -> None:
        # 這裡目前不需要狀態，先留空
        pass

    def update(self, env) -> None:
        # 球還沒發球就待在中間
        if not env.ball_in_play:
            target_x = (env.W - env.PADDLE_W) / 2
        else:
            ball_cx = env.ball_x + env.BALL_SIZE / 2
            target_x = ball_cx - env.PADDLE_W / 2

        # 以固定速度往 target_x 移動
        if target_x > env.p2_x + self.speed:
            env.p2_x += self.speed
        elif target_x < env.p2_x - self.speed:
            env.p2_x -= self.speed
        else:
            env.p2_x = target_x

        # 邊界 clamp
        if env.p2_x < 0:
            env.p2_x = 0
        if env.p2_x > env.W - env.PADDLE_W:
            env.p2_x = env.W - env.PADDLE_W
