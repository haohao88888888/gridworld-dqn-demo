# GridWorld-DQN 强化学习 Demo

这是一个基于 PyTorch 的小型强化学习项目：智能体在 GridWorld 网格地图中学习从起点移动到目标点，同时避开障碍物。项目使用 DQN（Deep Q-Network）算法。

## 项目亮点

- 自定义 GridWorld 环境，包含起点、目标点、障碍物、状态表示、动作空间和回合终止条件。
- 使用 PyTorch 实现 DQN 网络、经验回放、epsilon-greedy 探索策略和目标网络更新机制。
- 训练后自动保存模型、训练日志、奖励曲线、成功率曲线和一次贪心策略测试轨迹。
- 项目结构清晰，适合上传 GitHub，也适合写进简历项目经历。

## 项目结构

```text
gridworld-dqn-demo/
├─ README.md
├─ requirements.txt
├─ config.py
├─ train.py
├─ test.py
├─ envs/
│  ├─ __init__.py
│  └─ gridworld.py
├─ agents/
│  ├─ __init__.py
│  ├─ dqn_agent.py
│  └─ replay_buffer.py
├─ models/
│  ├─ __init__.py
│  └─ dqn.py
├─ utils/
│  ├─ __init__.py
│  ├─ plot.py
│  └─ seed.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  └─ figures/
└─ docs/
   └─ demo_notes.md
```

## 快速开始

```powershell
Set-Location D:\PythonProject\gridworld-dqn-demo
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py
.\.venv\Scripts\python test.py
```

如果只是想快速跑通流程，可以减少训练轮数：

```powershell
.\.venv\Scripts\python train.py --episodes 80
.\.venv\Scripts\python test.py --episodes 3
```

## 训练输出

训练完成后会生成：

```text
outputs/checkpoints/dqn_gridworld.pt      训练好的模型参数
outputs/logs/training_log.csv             每个 episode 的奖励、步数、成功情况
outputs/logs/training_config.json         本次训练配置
outputs/logs/greedy_path.txt              训练后贪心策略走出的路径
outputs/figures/training_curves.png       奖励、成功率、loss 曲线
```

## 算法简述

DQN 的核心思想是用神经网络近似 Q 函数：

```text
Q(state, action) ≈ 未来累计奖励
```

智能体每一步根据当前状态选择动作，上下左右移动。训练时使用 epsilon-greedy 策略：前期更多随机探索，后期更多选择当前 Q 值最高的动作。经验会被保存到 replay buffer 中，再随机采样 batch 训练网络，从而降低样本之间的相关性。

## 主要命令

```powershell
# 训练模型
python train.py

# 自定义训练轮数
python train.py --episodes 300

# 测试训练好的模型
python test.py

# 指定模型路径测试
python test.py --checkpoint outputs/checkpoints/dqn_gridworld.pt
```
