# 環境設定
1. python <= 3.12
2. uv
3. torch(cpu or gpu version) = 2.5.1+cu121 <- cu121 for gpu version
4. numpy = 2.3.3
5. pygame = 2.6.1
6. matplotlib = 3.10.7
# 參數調整
1. train_dqn.py -> def train -> env.mode (easy/hard)
2. train_dqn.py -> def train -> device (cuda or cpu)

# 使用者要求
1. 發球(左/右)
2. 在一般/高速情況下都可以穩定接球
3. 在高速下預判球路
4. 勝率達到九成以上

# to-do list
- [ ] 列出測試清單(單元測試以及整合測試)
- [ ] 完成架構圖
- [ ] 撰寫Readme/PRD
- [ ] 補上上次簡報缺失的內容
- [ ] DQN NN架構與內容


