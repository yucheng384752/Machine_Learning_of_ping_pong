# v0.0.2
# Changelog:
# - 新增 shaping_coef 參數，當球朝向玩家移動時，
#   會根據球與板子中心點的垂直距離給額外負向獎勵（dense reward）。
# - 調整預設獎勵參數為較小的 time penalty（但仍可在外部覆寫）。

import random
from typing import Tuple, Dict, Any, Optional

import numpy as np


class PongEnv:
    """
    簡易乒乓球環境：
    - 左邊板子：我們的 agent 控制
    - 右邊板子：簡單 rule-based AI（盯著球 y 走）
    - 動作空間：0=不動, 1=往上, 2=往下
    - 狀態空間：6 維連續向量 (float32)
        [ ball_x, ball_y, ball_vx, ball_vy, player_y, ai_y ]，均做 0~1 正規化
    """

    def __init__(
        self,
        max_steps: int = 1000,
        time_penalty: float = -0.005,
        hit_reward: float = 0.5,
        win_reward: float = 2.0,
        lose_penalty: float = -2.0,
        shaping_coef: float = 0.2,
        seed: Optional[int] = None,
    ):
        # 遊戲畫面邏輯解析度（與 game.py 一致）
        self.W, self.H = 160, 120

        # 幾何設定（與 game.py 一致）
        self.PADDLE_W, self.PADDLE_H = 4, 20
        self.BALL_SIZE = 4
        self.PADDLE_SPEED = 2
        self.BALL_SPEED = 2

        # 動作空間：0=stay, 1=up, 2=down
        self.action_space_n = 3

        # 獎勵設定
        self.max_steps = max_steps
        self.time_penalty = time_penalty
        self.hit_reward = hit_reward
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty
        self.shaping_coef = shaping_coef  # 新增：距離 shaping 強度

        # 內部狀態
        self.player_x = 10
        self.player_y = 0
        self.ai_x = self.W - 10 - self.PADDLE_W
        self.ai_y = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_vx = 0
        self.ball_vy = 0

        self.done = False
        self.steps = 0

        self._rng = random.Random()
        self.seed(seed)

    # -------- 公開 API --------
    def seed(self, seed: Optional[int] = None):
        """設定隨機種子（方便重現實驗）。"""
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)

    def reset(self) -> np.ndarray:
        """
        重置一局遊戲，回傳初始 state。
        state shape = (6,)
        """
        self.player_x = 10
        self.player_y = self.H // 2 - self.PADDLE_H // 2

        self.ai_x = self.W - 10 - self.PADDLE_W
        self.ai_y = self.H // 2 - self.PADDLE_H // 2

        self.ball_x = self.W // 2
        self.ball_y = self.H // 2

        self.ball_vx = self._rng.choice([-self.BALL_SPEED, self.BALL_SPEED])
        self.ball_vy = self._rng.choice([-self.BALL_SPEED, self.BALL_SPEED])

        self.done = False
        self.steps = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        執行一步環境更新。
        :param action: 0=不動, 1=往上, 2=往下
        :return: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() before step().")

        self.steps += 1

        # 1. 更新玩家板子（agent 控制）
        if action == 1:          # up
            self.player_y -= self.PADDLE_SPEED
        elif action == 2:        # down
            self.player_y += self.PADDLE_SPEED

        # 限制在場內
        self.player_y = max(0, min(self.H - self.PADDLE_H, self.player_y))

        # 2. 更新 AI 板子（簡單 rule-based：追著球 y）
        if self.ball_y < self.ai_y:
            self.ai_y -= self.PADDLE_SPEED
        elif self.ball_y > self.ai_y + self.PADDLE_H:
            self.ai_y += self.PADDLE_SPEED

        self.ai_y = max(0, min(self.H - self.PADDLE_H, self.ai_y))

        # 3. 更新球的位置
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # 基礎時間懲罰
        reward = self.time_penalty

        # 4. 碰到上下邊界，反彈
        if self.ball_y <= 0 or self.ball_y >= self.H - self.BALL_SIZE:
            self.ball_vy *= -1

        # 5. 撞到玩家板子
        if (
            self.player_x < self.ball_x < self.player_x + self.PADDLE_W
            and self.player_y < self.ball_y < self.player_y + self.PADDLE_H
        ):
            self.ball_vx *= -1
            reward += self.hit_reward

        # 6. 撞到 AI 板子
        if (
            self.ai_x < self.ball_x + self.BALL_SIZE < self.ai_x + self.PADDLE_W
            and self.ai_y < self.ball_y < self.ai_y + self.PADDLE_H
        ):
            self.ball_vx *= -1

        # 7. 狀態 shaping：當球朝向玩家移動時，根據垂直距離給額外負獎勵
        if self.shaping_coef > 0 and self.ball_vx < 0:
            ball_center_y = self.ball_y + self.BALL_SIZE / 2
            paddle_center_y = self.player_y + self.PADDLE_H / 2
            dist = abs(ball_center_y - paddle_center_y) / self.H  # 正規化距離 0~1
            reward -= self.shaping_coef * dist

        # 8. 出界判定（誰 miss 球）
        info: Dict[str, Any] = {}

        if self.ball_x < 0:  # 我方 miss
            reward += self.lose_penalty
            self.done = True
            info["result"] = "lose"
        elif self.ball_x > self.W:  # AI miss（我方得分）
            reward += self.win_reward
            self.done = True
            info["result"] = "win"

        # 9. 步數超過 max_steps 也結束（避免無限局）
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["result"] = info.get("result", "timeout")

        next_state = self._get_state()
        return next_state, float(reward), self.done, info

    # -------- 工具函式 --------
    def _get_state(self) -> np.ndarray:
        """
        把目前遊戲狀態轉成 0~1 的連續向量（方便丟進神經網路）。
        """
        # ball_x, ball_y → 0~1
        bx = self.ball_x / self.W
        by = self.ball_y / self.H

        # ball_vx, ball_vy → 0~1（把 [-BALL_SPEED, BALL_SPEED] 映射到 [0,1]）
        bvx = (self.ball_vx + self.BALL_SPEED) / (2 * self.BALL_SPEED)
        bvy = (self.ball_vy + self.BALL_SPEED) / (2 * self.BALL_SPEED)

        # player_y, ai_y → 0~1
        py = self.player_y / self.H
        ay = self.ai_y / self.H

        state = np.array([bx, by, bvx, bvy, py, ay], dtype=np.float32)
        return state
