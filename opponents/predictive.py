# opponents/predictive.py

from __future__ import annotations

from .base import OpponentBase


class PredictiveOpponent:
    """
    具備「預判」功能的 2P 對手：
      - 球往 2P 方向飛時，預測若干 frame 之後的球 x 位置
      - 依球速 (vy) 區分慢球/快球，對慢球縮短預測視窗，避免被「未來的牆反彈」騙走
      - 球沒往上飛時，退回中間待機
    """

    def __init__(
        self,
        speed: float = 7.0,
        reaction_delay: int = 3,
        anticipation: float = 4.0,
    ):
        self.speed = float(speed)
        self.reaction_delay = max(1, int(reaction_delay))
        self.anticipation = float(anticipation)

        self._frame_counter = 0

    def reset(self, env) -> None:
        self._frame_counter = 0

    def update(self, env) -> None:
        self._frame_counter += 1

        # 控制「反應頻率」：每 N frame 才動一次
        if self._frame_counter % self.reaction_delay != 0:
            return

        # 球往上（2P 方向）飛：防守模式
        if env.ball_in_play and env.ball_vy < 0:
            dy = (env.p2_y + env.PADDLE_H) - (env.ball_y + env.BALL_SIZE)
            if env.ball_vy != 0:
                frames_to_reach = abs(dy / env.ball_vy)
            else:
                frames_to_reach = 0.0

            # 用 env.initial_speed 當作球初速（你在 env_paia 裡有這個）
            speed_abs_vy = abs(env.ball_vy)
            slow_threshold = env.initial_speed * 1.2

            if speed_abs_vy < slow_threshold:
                # 慢球：縮短預測視窗，避免預測太遠
                effective_frames = min(frames_to_reach, 15.0)
                anticipate = max(1.0, self.anticipation * 0.5)
            else:
                # 快球：維持原先預判
                effective_frames = frames_to_reach
                anticipate = self.anticipation

            predicted_x = env.ball_x + env.ball_vx * (effective_frames + anticipate)
            # clamp 落點
            predicted_x = max(0, min(env.W - env.PADDLE_W, predicted_x))

            target_x = predicted_x
        else:
            # 球沒往上飛：回中間
            target_x = (env.W - env.PADDLE_W) / 2

        # 以有限速度往 target_x 移動
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
