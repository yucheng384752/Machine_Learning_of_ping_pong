# env_paia.py
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np


class PongEnvPAIA:
    """
    PAIA 版乒乓球環境（1P 為我們的 agent）：

    - 解析度：200 x 500
    - 1P：底部紅色板子，左右移動
    - 2P：上方藍色板子，rule-based AI
    - 動作空間（5 個）：
        0 = 不動
        1 = 左移
        2 = 右移
        3 = 發球向左
        4 = 發球向右

    - 狀態空間（10 維 float32）：
        [bx, by, bvx, bvy, p1x, p2x, ball_attached, we_serving, ox, oy]
      其中：
        bx, by: 球中心座標 / 場地尺寸 → [0,1]
        bvx, bvy: 球速度 / max_ball_speed 映射到 [0,1]
        p1x, p2x: 板子中心 x / 場地寬度 → [0,1]
        ball_attached: 1=球還沒發出、貼在發球方板子上；0=球在飛
        we_serving: 1=輪到 1P 發球；0=輪到 2P 發球（目前訓練中可都讓 1P 發）
        ox, oy: 障礙物中心座標 / 場地尺寸（easy 模式為 0）
    """

    def __init__(
        self,
        mode: str = "easy",       # "easy" or "hard"
        max_steps: int = 1000,
        time_penalty: float = -0.005,
        hit_reward: float = 0.5,
        win_reward: float = 2.0,
        lose_penalty: float = -2.0,
        max_ball_speed: float = 20.0,  # 用來做速度正規化與封頂
        seed: Optional[int] = None,
    ):
        self.mode = mode
        self.use_obstacle = (mode == "hard")

        # 場地設定（對齊 PAIA）
        self.W, self.H = 200, 500

        # 板子設定
        self.PADDLE_W, self.PADDLE_H = 40, 10
        self.PADDLE_SPEED = 5

        # 球設定
        self.BALL_SIZE = 10
        self.initial_speed = 7.0
        self.speed_increase_interval = 100  # 每 100 frame 速度 +1
        self.max_ball_speed = max_ball_speed

        # 動作空間：0~4
        self.action_space_n = 5

        # 獎勵參數
        self.max_steps = max_steps
        self.time_penalty = time_penalty
        self.hit_reward = hit_reward
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty

        # RNG
        self._rng = random.Random()
        self.seed(seed)

        # 狀態變數（reset 時會重新設定）
        self.steps = 0
        self.done = False

        self.p1_x = 0
        self.p1_y = 0
        self.p2_x = 0
        self.p2_y = 0

        self.p1_move_dir = 0  # -1,0,1：本 frame 1P 的水平移動方向（用於簡單切球）
        self.p2_move_dir = 0

        self.ball_x = 0
        self.ball_y = 0
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.ball_in_play = False      # 球是否已經發出
        self.we_serving = True         # 是否輪到 1P 發球
        self.frames_since_serve_prompt = 0
        self.frames_since_serve = 0    # 球發出後經過的 frame，用來加速

        # 障礙物（hard mode）
        self.obstacle_w = 30
        self.obstacle_h = 20
        self.obstacle_x = 0
        self.obstacle_y = 240
        self.obstacle_vx = 0.0

    # ----------- 公開 API -----------

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)

    def reset(self) -> np.ndarray:
        """
        開新局：設定板子位置、球貼在 1P 板子上等待發球。
        這裡每個 episode 視為「一球」，所以每次都由 1P 發球重新開始。
        """
        self.steps = 0
        self.done = False

        # 1P 初始：x=80, y=420
        self.p1_x = 80
        self.p1_y = 420

        # 2P 初始：x=80, y=70
        self.p2_x = 80
        self.p2_y = 70

        self.p1_move_dir = 0
        self.p2_move_dir = 0

        # 球貼在發球者板子上（先由 1P 開始）
        self.we_serving = True
        self.ball_in_play = False
        self.frames_since_serve_prompt = 0
        self.frames_since_serve = 0

        self._attach_ball_to_server()

        # hard 模式初始化障礙物
        if self.use_obstacle:
            xs = list(range(0, self.W - self.obstacle_w + 1, 20))
            self.obstacle_x = self._rng.choice(xs)
            self.obstacle_y = 240
            self.obstacle_vx = self._rng.choice([-5.0, 5.0])
        else:
            self.obstacle_x = 0
            self.obstacle_y = 0
            self.obstacle_vx = 0.0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        執行一步遊戲邏輯：
        - 根據 action 控制 1P
        - 更新 2P、球、障礙物
        - 回傳 (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() before step().")

        self.steps += 1
        reward = self.time_penalty  # 每一步基礎 time penalty
        info: Dict[str, Any] = {}

        # 1. 處理 1P 動作（左右移動 / 發球）
        self._update_player1(action)

        # 2. 處理 2P rule-based AI（例如追球 x）
        self._update_player2()

        # 3. 更新球 & 碰撞（含簡化版切球、邊界反彈、出局判定）
        ball_reward, result = self._update_ball()
        reward += ball_reward

        # 4. hard 模式更新障礙物
        if self.use_obstacle:
            self._update_obstacle()

        # 5. 依結果加上 win/lose 的獎勵
        if result == "win":
            reward += self.win_reward
            self.done = True
            info["result"] = "win"

        elif result == "lose":
            reward += self.lose_penalty

            # 額外：根據掉球當下，球與 1P 板子的水平距離，再多扣一些
            ball_cx = self.ball_x + self.BALL_SIZE / 2
            p1_cx = self.p1_x + self.PADDLE_W / 2

            dx_norm = abs(ball_cx - p1_cx) / self.W  # 0~1，左右對稱
            extra_penalty = 1.0 * dx_norm             # 這個 1.0 可以調大調小

            reward -= extra_penalty

            self.done = True
            info["result"] = "lose"
            info["miss_dx_norm"] = dx_norm  # 想 debug 可以順便存起來


        # 6. 時間結束條件（timeout）
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info.setdefault("result", "timeout")

        next_state = self._get_state()
        return next_state, float(reward), self.done, info

    # ----------- 內部邏輯 -----------

    def _attach_ball_to_server(self):
        """
        根據目前 we_serving 決定球的位置：
        - 若輪到 1P，球貼在 1P 板子上方
        - 若輪到 2P，可以貼在 2P 板子下方（目前訓練中主要用 1P）
        """
        if self.we_serving:
            self.ball_x = (
                self.p1_x + self.PADDLE_W // 2 - self.BALL_SIZE // 2
            )
            self.ball_y = self.p1_y - self.BALL_SIZE  # 貼在板子上方（往上打）
        else:
            self.ball_x = (
                self.p2_x + self.PADDLE_W // 2 - self.BALL_SIZE // 2
            )
            self.ball_y = self.p2_y + self.PADDLE_H   # 貼在 2P 板子下方（往下打）

        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def _serve_ball(self, direction: str):
        """
        發球：direction in {"left", "right"}
        初始速度為 self.initial_speed，往對手方向打。
        """
        speed = self.initial_speed

        if direction == "left":
            self.ball_vx = -speed
        else:
            self.ball_vx = speed

        # 1P 在下面 → 發球時 vy 要往上（負）；若是 2P 發球則 vy 往下（正）
        self.ball_vy = -speed if self.we_serving else speed
        self.ball_in_play = True
        self.frames_since_serve_prompt = 0
        self.frames_since_serve = 0

    def _update_player1(self, action: int):
        """
        根據 action 更新 1P 的 x & 發球（如果球還沒發出）。
        """
        # 記錄本 frame 的移動方向（用於簡單切球機制）
        move_dir = 0

        # 移動
        if action == 1:         # 左移
            self.p1_x -= self.PADDLE_SPEED
            move_dir = -1
        elif action == 2:       # 右移
            self.p1_x += self.PADDLE_SPEED
            move_dir = 1

        # 邊界限制
        self.p1_x = max(0, min(self.W - self.PADDLE_W, self.p1_x))
        self.p1_move_dir = move_dir

        # 發球（球尚未發出、且輪到 1P）
        if not self.ball_in_play and self.we_serving:
            if action == 3:
                self._serve_ball("left")
            elif action == 4:
                self._serve_ball("right")
            else:
                # 沒按發球鍵 → 球跟著板子走
                self._attach_ball_to_server()
                self.frames_since_serve_prompt += 1
                # 超過 150 frame 自動隨機發球
                if self.frames_since_serve_prompt >= 150:
                    direction = self._rng.choice(["left", "right"])
                    self._serve_ball(direction)

    def _update_player2(self):
        """
        簡單 rule-based AI：追著球的 x。
        """
        old_x = self.p2_x

        # 若球還沒發出，就追著中線或球的 x
        target_x = self.ball_x
        center_x = self.p2_x + self.PADDLE_W / 2

        if target_x < center_x:
            self.p2_x -= self.PADDLE_SPEED
        elif target_x > center_x:
            self.p2_x += self.PADDLE_SPEED

        self.p2_x = max(0, min(self.W - self.PADDLE_W, self.p2_x))

        dx = self.p2_x - old_x
        if dx < 0:
            self.p2_move_dir = -1
        elif dx > 0:
            self.p2_move_dir = 1
        else:
            self.p2_move_dir = 0

    def _increase_ball_speed_if_needed(self):
        """
        每隔一定步數增加球速（保持方向不變，只放大向量長度）。
        """
        self.frames_since_serve += 1

        if (
            self.frames_since_serve > 0
            and self.frames_since_serve % self.speed_increase_interval == 0
        ):
            vx, vy = self.ball_vx, self.ball_vy
            speed = (vx * vx + vy * vy) ** 0.5
            if speed <= 1e-6:
                return
            new_speed = min(self.max_ball_speed, speed + 1.0)
            scale = new_speed / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

    def _update_ball(self) -> Tuple[float, Optional[str]]:
        """
        更新球位置、處理邊界碰撞、板子碰撞、得分判定。
        回傳：
        - reward_delta: 這一步因為擊球等產生的額外獎勵
        - result: None / "win" / "lose"
        """
        if not self.ball_in_play:
            return 0.0, None

        reward_delta = 0.0
        result: Optional[str] = None

        # 1. 速度遞增（依 serve 後經過 frame 數）
        self._increase_ball_speed_if_needed()

        # 2. 更新位置
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # 球的 AABB
        ball_left = self.ball_x
        ball_right = self.ball_x + self.BALL_SIZE
        ball_top = self.ball_y
        ball_bottom = self.ball_y + self.BALL_SIZE

        # 3. 左右牆壁反彈
        if ball_left <= 0:
            self.ball_x = 0
            self.ball_vx = abs(self.ball_vx)  # 往右彈
            ball_left = self.ball_x
            ball_right = self.ball_x + self.BALL_SIZE
        elif ball_right >= self.W:
            self.ball_x = self.W - self.BALL_SIZE
            self.ball_vx = -abs(self.ball_vx)  # 往左彈
            ball_left = self.ball_x
            ball_right = self.ball_x + self.BALL_SIZE

        # 4. 上下邊界保護（理論上應該在出界前被 paddle 接到或 miss）
        if ball_top < 0:
            # 先不要立刻反彈，交由出界判定處理
            pass
        if ball_bottom > self.H:
            # 同上
            pass

        # 5. 撞到 1P 板子（在下面）
        p1_top = self.p1_y
        p1_bottom = self.p1_y + self.PADDLE_H
        p1_left = self.p1_x
        p1_right = self.p1_x + self.PADDLE_W

        if (
            ball_bottom >= p1_top
            and ball_top <= p1_bottom
            and ball_right >= p1_left
            and ball_left <= p1_right
            and self.ball_vy > 0  # 球往下飛才有可能被 1P 擋到
        ):
            self.ball_y = p1_top - self.BALL_SIZE
            ball_bottom = self.ball_y + self.BALL_SIZE

            self.ball_vy = -abs(self.ball_vy)

            # 簡化版切球
            move_dir = self.p1_move_dir  # -1,0,1
            if move_dir != 0:
                if self.ball_vx == 0.0:
                    self.ball_vx = move_dir * self.initial_speed
                else:
                    sign_vx = 1.0 if self.ball_vx > 0 else -1.0
                    if move_dir == sign_vx:
                        new_mag = min(
                            self.max_ball_speed, abs(self.ball_vx) + 3.0
                        )
                        self.ball_vx = sign_vx * new_mag
                    else:
                        self.ball_vx = -self.ball_vx

            reward_delta += self.hit_reward

        # 6. 撞到 2P 板子（在上面）
        p2_top = self.p2_y
        p2_bottom = self.p2_y + self.PADDLE_H
        p2_left = self.p2_x
        p2_right = self.p2_x + self.PADDLE_W

        if (
            ball_top <= p2_bottom
            and ball_bottom >= p2_top
            and ball_right >= p2_left
            and ball_left <= p2_right
            and self.ball_vy < 0  # 球往上飛
        ):
            self.ball_y = p2_bottom
            ball_top = self.ball_y
            self.ball_vy = abs(self.ball_vy)

        # 7. 出界判定（上下出界）
        if ball_bottom < 0:
            result = "win"
            self.ball_in_play = False
        elif ball_top > self.H:
            result = "lose"
            self.ball_in_play = False

        return reward_delta, result

    def _update_obstacle(self):
        """
        hard 模式障礙物左右移動、與球碰撞。
        球撞到障礙物時保持速度大小，只反轉垂直方向。
        """
        # 移動
        self.obstacle_x += self.obstacle_vx

        # 反彈
        if self.obstacle_x <= 0:
            self.obstacle_x = 0
            self.obstacle_vx = abs(self.obstacle_vx)
        elif self.obstacle_x + self.obstacle_w >= self.W:
            self.obstacle_x = self.W - self.obstacle_w
            self.obstacle_vx = -abs(self.obstacle_vx)

        if not self.ball_in_play:
            return

        # 球與障礙物 AABB 碰撞
        ball_left = self.ball_x
        ball_right = self.ball_x + self.BALL_SIZE
        ball_top = self.ball_y
        ball_bottom = self.ball_y + self.BALL_SIZE

        obs_left = self.obstacle_x
        obs_right = self.obstacle_x + self.obstacle_w
        obs_top = self.obstacle_y
        obs_bottom = self.obstacle_y + self.obstacle_h

        if (
            ball_right >= obs_left
            and ball_left <= obs_right
            and ball_bottom >= obs_top
            and ball_top <= obs_bottom
        ):
            # 決定從上還是從下撞到障礙物，簡化處理成改 vy
            if ball_bottom > obs_top and ball_top < obs_top:
                # 從上方撞到 → 往上彈
                self.ball_y = obs_top - self.BALL_SIZE - 1
                self.ball_vy = -abs(self.ball_vy)
            elif ball_top < obs_bottom and ball_bottom > obs_bottom:
                # 從下方撞到 → 往下彈
                self.ball_y = obs_bottom + 1
                self.ball_vy = abs(self.ball_vy)
            else:
                # 側面撞到就反轉 vx
                if self.ball_x < obs_left:
                    self.ball_x = obs_left - self.BALL_SIZE - 1
                    self.ball_vx = -abs(self.ball_vx)
                else:
                    self.ball_x = obs_right + 1
                    self.ball_vx = abs(self.ball_vx)

            # 速度大小維持不變（只改方向）

    # ----------- state 編碼 -----------

    def _get_state(self) -> np.ndarray:
        """
        將當前遊戲狀態轉為 10 維連續向量：
        [bx, by, bvx, bvy, p1x, p2x, ball_attached, we_serving, ox, oy]
        """
        # 球中心
        ball_cx = self.ball_x + self.BALL_SIZE / 2
        ball_cy = self.ball_y + self.BALL_SIZE / 2

        bx = ball_cx / self.W
        by = ball_cy / self.H

        # 速度正規化到 [0,1]（由 [-max_ball_speed, +max_ball_speed] 映射）
        max_v = self.max_ball_speed
        bvx = (self.ball_vx / max_v + 1.0) / 2.0
        bvy = (self.ball_vy / max_v + 1.0) / 2.0

        # 板子中心
        p1_center_x = self.p1_x + self.PADDLE_W / 2
        p2_center_x = self.p2_x + self.PADDLE_W / 2

        p1x = p1_center_x / self.W
        p2x = p2_center_x / self.W

        # 球是否仍「黏」在發球者板子上
        ball_attached = 0.0 if self.ball_in_play else 1.0

        we_serving = 1.0 if self.we_serving else 0.0

        # 障礙物位置（easy 模式就讓它們是 0）
        if self.use_obstacle:
            ox = (self.obstacle_x + self.obstacle_w / 2) / self.W
            oy = (self.obstacle_y + self.obstacle_h / 2) / self.H
        else:
            ox = 0.0
            oy = 0.0

        state = np.array(
            [bx, by, bvx, bvy, p1x, p2x, ball_attached, we_serving, ox, oy],
            dtype=np.float32,
        )
        return state
