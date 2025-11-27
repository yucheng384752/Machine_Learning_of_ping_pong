# 專案架構總覽（Breakdown）

本專案是一個「以 DQN 訓練 1P 乒乓球 AI，與 2P 規則型對手對戰」的完整系統，包含環境模擬、強化學習、對手 AI、視覺化遊玩與分析工具。
```
                   ┌─────────────────────────────────────────────┐
                   │       DQN-Based Pong Agent System            │
                   └─────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼────────────────────────────────────┐
       │                           │                                    │
┌───────────────────┐   ┌────────────────────────┐        ┌──────────────────────────┐
│  Environment       │   │  Reinforcement Learning │        │   Opponent AI (2P)       │
│ (env_paia.py)      │   │      (train_dqn.py)     │        │ (base / predictive / …)  │
└───────────────────┘   └────────────────────────┘        └──────────────────────────┘
       │                           │                                    │
       ▼                           ▼                                    ▼
- 球物理更新                - DQN 神經網路 (128-128 MLP)       - Simple-Follow AI
- 球反彈與速度調整         - Target Network Sync              - Predictive 落點預測AI
- 1P (Agent) 控制           - TD error 計算 & Hubor loss       - 隨球速調整追球速度
- 2P (Rule-Based) AI       - Prioritized Replay Buffer        - Degree 可獨立調整
- 障礙物（Hard mode）       - ε-greedy 探索                     - update(env) 方法
- Reward Shaping           - 訓練記錄 (loss/Q/θ/reward curve)
- State vector 提供給 DQN

                 ┌──────────────────────────────────────────┐
                 │  Visualization & Demo (play_paia_agent)  │
                 └──────────────────────────────────────────┘
                 - pygame 遊戲即時畫面
                 - 顯示：反彈次數、reward、epsilon、球速
                 - 左側資訊欄採等寬字體、左對齊
                 - 可載入 .pt 模型

                 ┌──────────────────────────────────────────┐
                 │        Analysis Tools (analysis/)        │
                 └──────────────────────────────────────────┘
                 - analyze_miss_positions：收集 1P 掉球位置
                 - plot_training_curves：產生 loss/Q/reward/θ 圖
                 - 可讀取 .npy 進行熱力圖 & 失誤分布
```
## 1. 遊戲環境（Environment Layer）

- 主要檔案：`PAIA/env_paia.py`
- 職責：
  - 建立乒乓球遊戲規則與物理：
    - 球的座標、速度、反彈（上/下邊界、左右牆）
    - 1P / 2P 板子位置與移動（每步 5px）
    - 出界、得分、回合輪流發球
  - 難度與模式：
    - `mode="normal"`：基本對戰
    - `mode="hard"`：加入障礙物（位置、大小、左右來回移動）
  - Reward shaping：
    - 時間步懲罰（鼓勵加快決策）
    - 與球水平距離的懲罰（鼓勵慢慢對齊來球）
    - 擊球成功、勝利與失敗的獎勵 / 懲罰
    - 快速球額外加成（high-speed bonus）
  - 狀態輸出（state vector）：
    - 球位置 / 速度
    - 1P / 2P 位置
    - 反彈次數、是否在高速區間等


## 2. 強化學習訓練（DQN Training Layer）

- 主要檔案：`PAIA/train_dqn.py`
- 職責：
  - DQN 網路：
    - 多層全連接 NN（輸入 state，輸出對每個 action 的 Q 值）
    - 使用 PyTorch 實作，支援 GPU/CPU 自動切換
  - 訓練流程：
    - `env.reset() → (state, action, reward, next_state, done)` 反覆互動
    - 記錄 transition 到 replay buffer
    - 從 buffer 取樣 mini-batch，計算 TD target
    - 以 loss function 更新 online network 參數 θ
    - 週期性更新 target network
  - 探索策略（ε-greedy）：
    - 起始 ε = 1.0，隨訓練步數逐漸 decay 到 ε_min
    - 控制貪心 vs 隨機探索比例
  - Replay Buffer：
    - 由 uniform sampling 更新為 Prioritized Replay（依 TD-error 給不同抽樣權重）
    - 提高「關鍵錯誤情節」的學習效率
  - 監控與輸出：
    - 儲存 `reward_curve.npy`, `loss_curve.npy`, `avg_max_q.npy`, `theta_norm.npy`
    - 定期儲存 `models/dqn_pong_best.pt`, `models/dqn_pong_last.pt`


## 3. 對手 AI 模組（Opponent AI Layer）

- 主要檔案：`PAIA/opponents/`
  - `base.py`：共用介面與抽象類別
  - `simple_follow.py`：單純追逐球的 x 位置
  - `predictive.py`：預測球落點再移動
- 職責：
  - 2P 的行為邏輯（非學習型）：
    - 根據球當前軌跡、速度進行追球
    - 可透過 difficulty 調整速度與反應延遲
  - 提供「固定但可調強度」的 sparring partner：
    - easy / normal / hard 三種等級
    - 用來測試 1P DQN 在不同對手下的穩定度


## 4. 視覺化與遊玩介面（Play & Visualization Layer）

- 主要檔案：`PAIA/play_paia_agent.py`
- 職責：
  - 載入訓練完成的 DQN 模型（`.pt` 檔）
  - 建立與 `PongEnvPAIA` 的互動 loop：
    - `state → model(action) → env.step(action)`
  - 使用 `pygame` 顯示：
    - 球、板子、障礙物
    - 反彈次數
    - 當前 reward
    - win/lose 統計
  - 視覺化 debug：
    - 畫面放大（像素風格）
    - UI 文字左對齊顯示，便於錄影與 demo


## 5. 分析與統計工具（Analysis Layer）

- 主要檔案：`analysis/`
  - `analyze_miss_positions.py`：
    - 重複載入模型與環境，自動遊玩多回合
    - 蒐集掉球位置（Left / Middle / Right）分布
    - 輸出統計結果與 `.npy` 檔案
  - `plot_training_curves.py`：
    - 讀取訓練過程產生的 `.npy` 檔
    - 畫出：
      - reward 曲線
      - loss 曲線
      - avg/max Q 曲線
      - θ-norm 曲線
- 職責：
  - 幫助分析：
    - 模型是否過度偏向某一側（例如右側漏球）
    - 在高速球階段是否有明顯失誤集中區
    - 訓練是否收斂或發散
  - 支援報告與簡報使用：
    - 直接產生可貼進投影片的圖


## 6. 測試與專案管理（Testing & Management Layer）

- 測試層級：
  - 單元測試（Unit Test）：
    - 環境 reset、ball update、paddle move、obstacle 行為
    - DQN forward / loss / replay buffer
    - opponent AI 反應與速度
  - 整合測試（Integration Test）：
    - env + DQN 跑 1000 steps 不 crash
    - play_paia_agent 實際遊玩
    - hard 模式下障礙物存在且會影響球路徑
- 專案管理重點：
  - 功能完成度：DQN 訓練 / 2P 模組化 / 視覺化 / hard-mode
  - 效能指標：勝率、平均反彈次數、loss / Q 曲線
  - 驗收條件：
    - normal 模式對 simple-follow 有穩定優勢
    - hard 模式可應付多次高速反彈
    - lose position 有足夠樣本可分析

# 專案管理
- 功能清單：
  - DQN訓練
  - 2P AI 模組化
  - Reward shaping
  - Hard-mode 障礙物
  - 視覺化展示
- 效能指標：
  - Reward_curve.npy：訓練成果
  - Loss_curve.npy：模型穩定度
  - Avg_max_q.npy：policy改善
  - Theta_norm.npy：是否爆炸
- 介面(pygame)：
  - 原畫面放大2倍
  - Debug 資訊顯示
- 驗收： 
  - Agent在easy模式擊敗simple-follow
  - Agent在hard模式保持90%勝率
  - Lose position 至少收集到>50筆資料
  - 障礙物能正確迴避
  - 2P 兩個難度模式的速度都正確
    
# 系統分析
- 遊戲環境
  - 檔案：env_paia.py
  - 包含：
    - 球物理設定(反彈、速度、出局)
    - 玩家移動(1P使用DQN，2P用AI)
    - Hard模式障礙物設定
    - Reward shaping
    - State vector 最終輸出到DQN
- 強化學習
  - 檔案：train_dqn.py
  - 包含：
    - DQN network
    - Replay buffer
    - TD target 計算
    - Target network 更新
    - ϵ-greedy 探索
    - 四個監控指標：reward curve、loss curve 、avg max Q、 θ-norm
- 對手AI
  - 檔案：opponents/
  - 包含：
    - Simple_follow(直線跟球)
    - Predictive(預測球的落點)
    - 速度與反應延遲由difficulty控制
- 視覺化遊玩
  - 檔案：play_paia_agent.py
  - 包含：
    - Pygame介面
    - 模型載入
    - Agent vs 2P 對戰顯示
      
# 環境設定
- 難度與模式
- <img width="871" height="163" alt="image" src="https://github.com/user-attachments/assets/a6222ba4-8524-49b9-8b61-a74faad2661e" />
- DQN設定
  - 𝛾 = 0.99
  - Buffer = 50k
  - Batch size = 64
  - Target update = 1000 steps
  - Epsilon: 1 → 0.05(5000 decay steps)
- Reward 設計
    - 時間步：-0.01
    - 跟球對齊：-0.02*𝑑𝑥
    - 擊球成功：小幅正 reward
    - Win：+1.5
    - Lose：-3 ~ -4(依照𝑑𝑥調整)
    - 可加成：快速擊球 bouns
      
# 測試清單
- 單元測試
  - 環境測試
    - Test_reset：確保環境初始化
    - Test_update_ball：人工指定球的位置與速度，測試球的碰撞、邊界、出局
    - Test_update_player1：測試player的發球與動作
    - Test_update_player2：確認2P AI速度變化
    - Test_state_feature：確保state維度與數值範圍正確
    - Test_obstacle：hard 模式下，障礙物的位置、移動、碰撞
  - DQN模型測試
    - Test_dqn_forward：測試丟入隨機state，輸出的shape是否正確
    - Test_replay_buffer：push+sample，檢查shape是否正確
    - Test_loss_target_calc：手動產生batch，檢查TD target是否正確
    - Test_epsilon_schedule：測試epsilon是否從1 => 0.05線性下降
    - Test_parameter_update：測試單步backward，確認θ-norm變化
  - 2P對手測試
    - Test_opponent_update：呼叫update(env)，觀察P2是否向球靠近
    - Test_opponent_speed：在 easy/normal/hard要有三種速度
    - Test_predictive_ai：測試預測球落點是否正確
- 整合測試
  - Test_episode_run：env.reset → 500steps → 確保不會crash
  - Test_train_100_steps：執行100 steps DQN，𝜃 會更新
  - Test_play_agent：使用play_paia_agent實際跑一分鐘
  - Test_hard_mode_with_obstacle：hard模式下，障礙物存在且會阻擋球
  - Test_loss_position_analysis：測試analyze_miss_position能夠蒐集lose資料

# to-do list
- [X] 列出測試清單(單元測試以及整合測試)
- [X] 完成架構圖
- [X] 撰寫Readme
- [X] 補上breakdown
- [X] 補上上次簡報缺失的內容
- [X] DQN NN架構與內容
- [X] 補上loss function輪廓圖
- [X] Q learning 輸出說明

# 網頁連結
- https://app.paia-arena.com/zh/game/3



