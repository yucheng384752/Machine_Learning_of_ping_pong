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
