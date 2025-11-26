"""
PAIA 版本的乒乓球強化學習環境

- 螢幕大小: 200 x 500
- 1P: 紅色板子，在下方 (初始 (80, 420))
- 2P: 藍色板子，在上方 (初始 (80, 70))
- 球: 綠色方塊 10 x 10
- hard 模式: 中間加入黃色移動障礙物 30 x 20

狀態向量 (10 維, 皆為 0~1 正規化):
    [0] bx       球 x 座標 / W
    [1] by       球 y 座標 / H
    [2] bvx      球 vx 速度, 映射到 [0,1]
    [3] bvy      球 vy 速度, 映射到 [0,1]
    [4] p1x      1P 板子 x / W
    [5] p2x      2P 板子 x / W
    [6] attached 球是否黏在板子上 (0 or 1)
    [7] we_serving 是否 1P 發球輪到 (0 or 1)
    [8] ox       (hard 模式) 障礙物 x / W, 若無障礙物則 0
    [9] oy       (hard 模式) 障礙物 y / H, 若無障礙物則 0

step(action) 輸入動作:
    0: 不動 / 不發球
    1: 1P 往左移動
    2: 1P 往右移動
    3: 1P 往左發球  (若目前輪到 1P 發球且球未在場上)
    4: 1P 往右發球  (同上)

回傳:
    next_state: np.ndarray, shape=(10,)
    reward: float
    done: bool
    info: dict, 可能包含
        - "result": "win" / "lose" / "timeout"
        - "miss_dx_norm": 掉球時球心與板子中心的水平距離(0~1)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PongEnvPAIA:
    # 幾何設定
    W: int = 200
    H: int = 500

    PADDLE_W: int = 40
    PADDLE_H: int = 10
    BALL_SIZE: int = 10

    OBSTACLE_W: int = 30
    OBSTACLE_H: int = 20

    # 速度設定
    PADDLE_SPEED: int = 5
    BALL_INIT_SPEED: float = 7.0
    BALL_SPEED_INC_INTERVAL: int = 100  # 每 100 frame 加速一次
    BALL_SPEED_INC: float = 1.0
    BALL_SPEED_MAX: float = 18.0

    OBSTACLE_SPEED: int = 5

    # 強化學習相關 reward 參數
    time_penalty: float = -0.01      # 每步輕微懲罰，避免拖時間
    hit_reward: float = 0.15         # 每次成功由 1P 擊球的 reward
    win_reward: float = 2.0          # 1P 得分
    lose_penalty: float = -3.0       # 1P 掉球
    high_speed_bonus_scale: float = 0.03  # 球速越快，擊球時多給一點 bonus

    def __init__(
        self,
        mode: str = "easy",
        max_steps: int = 2000,
        seed: Optional[int] = None,
    ) -> None:
        assert mode in ("easy", "hard")
        self.mode = mode
        self.use_obstacle: bool = mode == "hard"
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 遊戲狀態變數
        self.steps: int = 0
        self.done: bool = False

        # 1P, 2P 位置
        self.p1_x: float = 0.0
        self.p1_y: float = 0.0
        self.p2_x: float = 0.0
        self.p2_y: float = 0.0

        # 上一 frame 板子位移 (用於切球判定)
        self.p1_last_dx: float = 0.0
        self.p2_last_dx: float = 0.0

        # 球狀態
        self.ball_x: float = 0.0
        self.ball_y: float = 0.0
        self.ball_vx: float = 0.0
        self.ball_vy: float = 0.0
        self.ball_in_play: bool = False  # True 代表球正在場上飛行
        self.we_serving: bool = True     # True = 1P 發球, False = 2P 發球
        self.ball_speed_timer: int = 0   # 用來計算何時加速

        # hard 模式障礙物
        self.obstacle_x: float = 0.0
        self.obstacle_y: float = 0.0
        self.obstacle_vx: float = 0.0
        
        self.ACTIONS = [0, 1, 2, 3, 4]
        self.action_space_n = len(self.ACTIONS)
        
        # 2P AI 參數（藍板）
        self.p2_speed = 5.0
        self.p2_reaction_delay = 3
        self.p2_anticipation = 4.0

        # 紀錄 1P / 2P 當前移動方向與上一 frame 的位移量
        self.p1_move_dir = 0      # -1 / 0 / 1
        self.p2_move_dir = 0
        self.p1_last_dx = 0.0
        self.p2_last_dx = 0.0
        
        
        # 對戰 AI（2P）：預設沒有對手，由外部指定
        self.opponent: Optional[object] = None

        self.reset()

    # --------------------------------------------------------------------- #
    # 公開 API
    # --------------------------------------------------------------------- #

    def reset(self) -> np.ndarray:
        """重置一局遊戲，並回傳初始 state。"""

        self.steps = 0
        self.done = False

        # 1P 在下方 (80, 420)
        self.p1_x = 80.0
        self.p1_y = 420.0

        # 2P 在上方 (80, 70)
        self.p2_x = 80.0
        self.p2_y = 70.0

        self.p1_last_dx = 0.0
        self.p2_last_dx = 0.0

        # 每局一開始固定輪到 1P 發球
        self.we_serving = True
        self.ball_in_play = False

        # 球貼在 1P 板子中心
        self._attach_ball_to_p1()

        self.ball_speed_timer = 0

        # hard 模式: 初始化障礙物
        if self.use_obstacle:
            possible_x = list(range(0, self.W - self.OBSTACLE_W + 1, 20))
            self.obstacle_x = float(random.choice(possible_x))
            self.obstacle_y = 240.0
            self.obstacle_vx = float(random.choice([-self.OBSTACLE_SPEED, self.OBSTACLE_SPEED]))
        else:
            self.obstacle_x = 0.0
            self.obstacle_y = 0.0
            self.obstacle_vx = 0.0

            
        # 移動相關
        self.p1_move_dir = 0
        self.p2_move_dir = 0
        self.p1_last_dx = 0.0
        self.p2_last_dx = 0.0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        執行一步遊戲邏輯:
        - 根據 action 控制 1P
        - 2P 使用簡單 rule-based AI
        - 更新球、障礙物、計算 reward

        action:
            0: 不動
            1: 左移
            2: 右移
            3: 往左發球
            4: 往右發球
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() before step().")

        self.steps += 1
        reward = self.time_penalty  # 基本時間懲罰
        info: Dict[str, Any] = {}

        # 1. 處理 1P 動作
        self._update_player1(action)

        # 2. 2P rule-based AI (追球)
        self._update_player2()

        # 3. 更新球 + 物理 + 回傳球相關 reward、結果
        ball_reward, result = self._update_ball()
        reward += ball_reward

        # 4. hard 模式障礙物
        if self.use_obstacle:
            self._update_obstacle()

        # 5. 對齊球的提前跑位 reward shaping
        #    只有在球往 1P 飛 (vy > 0) 且球在場上時才計算
        if self.ball_in_play and self.ball_vy > 0:
            ball_cx = self.ball_x + self.BALL_SIZE / 2
            p1_cx = self.p1_x + self.PADDLE_W / 2
            dx_norm = abs(ball_cx - p1_cx) / self.W  # 0~1
            # 越靠近球越好 → 懲罰越小
            reward -= 0.02 * dx_norm

        # 6. 依結果增加 win/lose 獎勵
        if result == "win":
            reward += self.win_reward
            self.done = True
            info["result"] = "win"

        elif result == "lose":
            reward += self.lose_penalty

            ball_cx = self.ball_x + self.BALL_SIZE / 2
            p1_cx = self.p1_x + self.PADDLE_W / 2
            dx_norm = abs(ball_cx - p1_cx) / self.W

            # 根據 miss 的水平距離給額外懲罰
            extra_penalty = 1.5 * dx_norm
            reward -= extra_penalty

            self.done = True
            info["result"] = "lose"
            info["miss_dx_norm"] = dx_norm

        # 7. timeout 條件
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info.setdefault("result", "timeout")

        next_state = self._get_state()
        return next_state, float(reward), self.done, info

    # --------------------------------------------------------------------- #
    # 內部輔助函式
    # --------------------------------------------------------------------- #

    def _attach_ball_to_p1(self) -> None:
        """讓球貼在 1P 板子上 (發球前狀態)."""
        self.ball_x = self.p1_x + (self.PADDLE_W - self.BALL_SIZE) / 2
        self.ball_y = self.p1_y - self.BALL_SIZE
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_in_play = False

    def _attach_ball_to_p2(self) -> None:
        """讓球貼在 2P 板子上 (2P 發球前狀態)."""
        self.ball_x = self.p2_x + (self.PADDLE_W - self.BALL_SIZE) / 2
        self.ball_y = self.p2_y + self.PADDLE_H
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_in_play = False

    def _update_player1(self, action: int) -> None:
        """根據 action 更新 1P 板子位置或發球。"""
        old_x = self.p1_x

        if action == 1:  # left
            self.p1_x -= self.PADDLE_SPEED
        elif action == 2:  # right
            self.p1_x += self.PADDLE_SPEED
        self.p1_x = max(0.0, min(self.p1_x, self.W - self.PADDLE_W))

        # 發球: 只有輪到 1P 發球且球不在場上才有效
        if self.we_serving and not self.ball_in_play:
            if action == 3:  # 往左發球
                self._serve_from_p1(direction=-1)
            elif action == 4:  # 往右發球
                self._serve_from_p1(direction=+1)

        self.p1_last_dx = self.p1_x - old_x

    def _serve_from_p1(self, direction: int) -> None:
        """1P 發球，direction = -1(往左) 或 +1(往右)."""
        self.ball_in_play = True
        self.ball_speed_timer = 0

        speed = self.BALL_INIT_SPEED
        # 用斜向, vx 看左右, vy 一律往上
        self.ball_vx = direction * speed / math.sqrt(2)
        self.ball_vy = -speed / math.sqrt(2)

        # 球從板子中央發出
        self.ball_x = self.p1_x + (self.PADDLE_W - self.BALL_SIZE) / 2
        self.ball_y = self.p1_y - self.BALL_SIZE

    def _serve_from_p2(self) -> None:
        """2P 發球 (簡化: 用隨機左右方向)."""
        self.ball_in_play = True
        self.ball_speed_timer = 0

        direction = random.choice([-1, 1])
        speed = self.BALL_INIT_SPEED

        self.ball_vx = direction * speed / math.sqrt(2)
        self.ball_vy = speed / math.sqrt(2)  # 往下

        self.ball_x = self.p2_x + (self.PADDLE_W - self.BALL_SIZE) / 2
        self.ball_y = self.p2_y + self.PADDLE_H

    def _update_player2(self) -> None:
        """更新 2P（藍板）位置，並記錄移動方向與位移量。"""
        old_x = self.p2_x

        if hasattr(self, "opponent") and self.opponent is not None:
            # 使用 opponents 模組提供的 AI
            self.opponent.update(self)
        else:
            # fallback：簡單 rule-based 追球
            if not self.ball_in_play:
                return

            ball_cx = self.ball_x + self.BALL_SIZE / 2
            p2_cx = self.p2_x + self.PADDLE_W / 2

            # 只在球往上飛時跟球，避免亂跑
            if self.ball_vy < 0:
                if ball_cx < p2_cx - 2:
                    self.p2_x -= self.p2_speed
                elif ball_cx > p2_cx + 2:
                    self.p2_x += self.p2_speed

            # 邊界限制
            self.p2_x = max(0, min(self.p2_x, self.W - self.PADDLE_W))

        # ---- 記錄這一 frame 的結果 ----
        dx = self.p2_x - old_x
        self.p2_last_dx = dx
        if dx > 0:
            self.p2_move_dir = 1
        elif dx < 0:
            self.p2_move_dir = -1
        else:
            self.p2_move_dir = 0


        self.p2_x = max(0.0, min(self.p2_x, self.W - self.PADDLE_W))
        self.p2_last_dx = self.p2_x - old_x

    def _update_ball(self) -> Tuple[float, Optional[str]]:
        """
        更新球位置 / 速度，處理所有碰撞，回傳 (球相關 reward, 結果)
        結果: None / "win" / "lose"
        """
        reward = 0.0
        result: Optional[str] = None

        # 如果目前沒有球在場上，直接讓球貼在發球方板子上
        if not self.ball_in_play:
            if self.we_serving:
                self._attach_ball_to_p1()
            else:
                self._attach_ball_to_p2()
            return reward, None

        # --- 1. 球速度隨時間加速 ---
        self.ball_speed_timer += 1
        if self.ball_speed_timer % self.BALL_SPEED_INC_INTERVAL == 0:
            # 調整球速大小，但方向不變
            vx, vy = self.ball_vx, self.ball_vy
            speed = math.sqrt(vx * vx + vy * vy)
            new_speed = min(speed + self.BALL_SPEED_INC, self.BALL_SPEED_MAX)
            if speed > 1e-6:
                scale = new_speed / speed
                self.ball_vx *= scale
                self.ball_vy *= scale

        # --- 2. 更新球的位置 ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # --- 3. 撞左右牆 (反彈) ---
        if self.ball_x <= 0:
            self.ball_x = 0
            self.ball_vx *= -1
        elif self.ball_x + self.BALL_SIZE >= self.W:
            self.ball_x = self.W - self.BALL_SIZE
            self.ball_vx *= -1

        # --- 4. 上下邊界 → win / lose ---
        if self.ball_y <= 0:
            # 球從上方飛出去 → 1P 得分
            result = "win"
            self.ball_in_play = False
            # 下一回合換 2P 發球
            self.we_serving = False
            return reward, result

        if self.ball_y + self.BALL_SIZE >= self.H:
            # 球從下方飛出去 → 1P 掉球
            result = "lose"
            self.ball_in_play = False
            self.we_serving = True  # 下一球換 1P 發球
            return reward, result

        # --- 5. 與 1P / 2P 碰撞 ---
        reward += self._handle_paddle_collision()

        # --- 6. hard 模式: 與障礙物碰撞 ---
        if self.use_obstacle:
            self._handle_obstacle_collision()

        return reward, result

    def _handle_paddle_collision(self) -> float:
        """處理與兩個板子的碰撞，並回傳與 1P 擊球相關的 reward。"""
        reward = 0.0

        ball_rect = (
            self.ball_x,
            self.ball_y,
            self.BALL_SIZE,
            self.BALL_SIZE,
        )

        p1_rect = (self.p1_x, self.p1_y, self.PADDLE_W, self.PADDLE_H)
        p2_rect = (self.p2_x, self.p2_y, self.PADDLE_W, self.PADDLE_H)

        def rect_overlap(a, b) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            return not (
                ax + aw < bx
                or bx + bw < ax
                or ay + ah < by
                or by + bh < ay
            )

        # ---- 與 1P 碰撞 (球往下飛時) ----
        if self.ball_vy > 0 and rect_overlap(ball_rect, p1_rect):
            # 把球調整到板子上面，避免卡在裡面
            self.ball_y = self.p1_y - self.BALL_SIZE
            self.ball_vy = -abs(self.ball_vy)

            # 切球機制:
            # 1. 板子與球水平移動方向相同 → vx 增加 3
            # 2. 相反 → 直接反轉 vx
            # 3. 板子靜止 → 只做一般反彈 (vx 不變)
            if abs(self.p1_last_dx) > 1e-6:
                if self.p1_last_dx * self.ball_vx > 0:
                    # 同方向
                    self.ball_vx += 3.0 * (1 if self.ball_vx >= 0 else -1)
                else:
                    # 反方向
                    self.ball_vx *= -1

            # 計算球速，用來給 high-speed bonus
            speed = math.sqrt(self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy)
            speed_bonus = max(0.0, (speed - self.BALL_INIT_SPEED)) * self.high_speed_bonus_scale

            reward += self.hit_reward + speed_bonus

        # ---- 與 2P 碰撞 (球往上飛時) ----
        if self.ball_vy < 0 and rect_overlap(ball_rect, p2_rect):
            self.ball_y = self.p2_y + self.PADDLE_H
            self.ball_vy = abs(self.ball_vy)

            # 2P 也依照板子移動方向做切球，但這不給 reward，只影響軌跡
            if abs(self.p2_last_dx) > 1e-6:
                if self.p2_last_dx * self.ball_vx > 0:
                    self.ball_vx += 3.0 * (1 if self.ball_vx >= 0 else -1)
                else:
                    self.ball_vx *= -1

        return reward

    def _handle_obstacle_collision(self) -> None:
        """球與障礙物的 AABB + 簡單反彈處理。"""
        if not self.use_obstacle:
            return

        obs_left = self.obstacle_x
        obs_right = self.obstacle_x + self.OBSTACLE_W
        obs_top = self.obstacle_y
        obs_bottom = self.obstacle_y + self.OBSTACLE_H


        ball_left = self.ball_x
        ball_right = self.ball_x + self.BALL_SIZE
        ball_top = self.ball_y
        ball_bottom = self.ball_y + self.BALL_SIZE

        if (
            ball_right >= obs_left
            and ball_left <= obs_right
            and ball_bottom >= obs_top
            and ball_top <= obs_bottom
        ):
            # 計算哪一個方向重疊比較小，當成撞擊面
            overlap_left = ball_right - obs_left
            overlap_right = obs_right - ball_left
            overlap_top = ball_bottom - obs_top
            overlap_bottom = obs_bottom - ball_top

            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if min_overlap == overlap_top:
                # 從上方撞到：往上彈回
                self.ball_y = obs_top - self.BALL_SIZE - 1
                self.ball_vy = -abs(self.ball_vy)
            elif min_overlap == overlap_bottom:
                # 從下方撞到：往下彈回
                self.ball_y = obs_bottom + 1
                self.ball_vy = abs(self.ball_vy)
            elif min_overlap == overlap_left:
                # 從左邊撞到：往左彈
                self.ball_x = obs_left - self.BALL_SIZE - 1
                self.ball_vx = -abs(self.ball_vx)
            else:
                # 從右邊撞到：往右彈
                self.ball_x = obs_right + 1
                self.ball_vx = abs(self.ball_vx)


    def _update_obstacle(self) -> None:
        """hard 模式下，讓障礙物左右移動並在邊界反彈。"""
        if not self.use_obstacle:
            return

        self.obstacle_x += self.obstacle_vx

        if self.obstacle_x <= 0:
            self.obstacle_x = 0
            self.obstacle_vx = abs(self.obstacle_vx)
        elif self.obstacle_x + self.OBSTACLE_W >= self.W:
            self.obstacle_x = self.W - self.OBSTACLE_W
            self.obstacle_vx = -abs(self.obstacle_vx)



    # --------------------------------------------------------------------- #
    # state 轉換
    # --------------------------------------------------------------------- #

    def _get_state(self) -> np.ndarray:
        """把目前 internal 狀態轉成 10 維的 RL state。"""
        # 位置正規化
        bx_norm = self.ball_x / self.W
        by_norm = self.ball_y / self.H
        p1x_norm = self.p1_x / self.W
        p2x_norm = self.p2_x / self.W

        # 速度正規化: 把 [-Vmax, Vmax] 映射到 [0,1]
        vmax = max(self.BALL_SPEED_MAX, self.BALL_INIT_SPEED + 6.0)
        bvx_norm = (self.ball_vx + vmax) / (2 * vmax)
        bvy_norm = (self.ball_vy + vmax) / (2 * vmax)

        # 球是否貼板子、是否輪到我們發球
        attached = 0.0 if self.ball_in_play else 1.0
        serving = 1.0 if self.we_serving else 0.0

        if self.use_obstacle:
            ox_norm = self.obstacle_x / self.W
            oy_norm = self.obstacle_y / self.H
        else:
            ox_norm = 0.0
            oy_norm = 0.0

        state = np.array(
            [
                bx_norm,
                by_norm,
                bvx_norm,
                bvy_norm,
                p1x_norm,
                p2x_norm,
                attached,
                serving,
                ox_norm,
                oy_norm,
            ],
            dtype=np.float32,
        )

        return state
