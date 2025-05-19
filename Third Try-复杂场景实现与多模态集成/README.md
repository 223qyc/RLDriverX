# 智能小车强化学习项目

这是一个基于强化学习的2D自动驾驶项目，用于训练智能小车在复杂环境中寻找动态目标。项目实现了更真实的物理模型、动态目标移动和丰富的可视化功能。

## 项目特点

- 2D环境中的自动驾驶小车
- 真实物理模型（惯性、加速度、摩擦力、转向力学）
- 多模态感知系统（雷达和视觉）
- 动态和静态障碍物
- 平滑移动的动态目标（非瞬移）
- 高级强化学习算法
- 美观的可视化系统和详细数据记录
- 多种统计分析和热图生成
- 高质量视频录制功能

## 环境要求

- Python 3.8+
- PyTorch
- Gymnasium
- OpenCV
- Matplotlib
- 其他依赖见requirements.txt

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
.
├── src/                 # 源代码目录
│   ├── agent/           # 智能体模块
│   │   ├── __init__.py  # 初始化文件
│   │   └── agent.py     # 智能体实现，包含DDPG算法
│   ├── config/          # 配置文件
│   │   ├── __init__.py  # 初始化文件
│   │   └── environment_config.py  # 环境、障碍物、奖励等配置
│   ├── environment/     # 环境模块
│   │   ├── __init__.py  # 初始化文件
│   │   ├── environment.py  # 小车环境实现
│   │   └── geometry.py     # 几何计算工具
│   ├── models/          # 模型模块
│   │   ├── __init__.py  # 初始化文件
│   │   └── network.py   # 神经网络模型结构
│   ├── utils/           # 工具模块
│   │   ├── __init__.py  # 初始化文件
│   │   └── metrics.py   # 指标记录和分析
│   ├── visualization/   # 可视化模块
│   │   ├── __init__.py  # 初始化文件
│   │   └── visualizer.py  # 可视化工具
│   ├── train.py         # 训练主脚本
│   ├── test.py          # 测试与评估脚本
│   └── test_environment.py # 环境测试脚本
├── logs/                # 日志和结果保存目录
└── requirements.txt     # 项目依赖
```

## 核心文件功能

### 环境模块 (`src/environment/`)

- **environment.py**: 核心环境类，实现了:
  - 真实物理模型（速度、转向、惯性、摩擦力）
  - 雷达与视觉感知系统
  - 动态和静态障碍物管理
  - 动态目标管理
  - 碰撞检测
  - 奖励计算
  - 渲染功能

- **geometry.py**: 提供几何相关的计算工具:
  - 射线与圆相交计算
  - 碰撞检测算法
  - 获取车辆角点坐标

### 智能体模块 (`src/agent/`)

- **agent.py**: 实现强化学习智能体:
  - 基于DDPG (Deep Deterministic Policy Gradient)算法
  - 使用Actor-Critic网络结构
  - 经验回放缓冲区
  - 目标网络软更新
  - 探索噪声管理
  - 模型保存与加载

### 模型模块 (`src/models/`)

- **network.py**: 定义神经网络结构:
  - 多模态输入处理（雷达和视觉）
  - 卷积网络处理视觉输入
  - 全连接网络处理雷达输入
  - 融合网络整合多模态信息
  - Actor和Critic网络

### 配置模块 (`src/config/`)

- **environment_config.py**: 存储所有环境配置:
  - 环境基本参数（尺寸、小车参数、物理参数等）
  - 障碍物参数（数量、大小、速度等）
  - 目标参数（大小、移动特性等）
  - 奖励参数（各类奖励和惩罚权重）
  - 渲染参数（颜色、大小、可视化选项）

### 工具模块 (`src/utils/`)

- **metrics.py**: 提供指标记录和分析功能:
  - 训练过程指标记录
  - 性能评估
  - 数据可视化
  - 统计分析
  - 热图生成

### 可视化模块 (`src/visualization/`)

- **visualizer.py**: 负责环境可视化:
  - 渲染环境状态
  - 绘制车辆、障碍物和目标
  - 显示雷达、视觉和轨迹
  - 添加信息面板
  - 生成热图

### 训练和测试脚本

- **train.py**: 训练主脚本:
  - 处理命令行参数
  - 设置训练环境
  - 创建和训练智能体
  - 记录训练指标
  - 生成训练视频
  - 保存训练模型

- **test.py**: 测试脚本:
  - 评估训练模型
  - 生成详细的性能分析
  - 录制高质量演示视频
  - 创建热图和统计图表

## 命令行参数说明

### 训练命令参数 (`train.py`)

| 参数               | 类型      | 默认值   | 说明                                    |
|--------------------|-----------|----------|----------------------------------------|
| `--num_episodes`   | int       | 1000     | 训练总回合数                           |
| `--max_steps`      | int       | 500      | 每个回合的最大步数                     |
| `--eval_interval`  | int       | 10       | 评估间隔（回合数）                     |
| `--save_interval`  | int       | 100      | 模型保存间隔（回合数）                 |
| `--learning_rate`  | float     | 3e-4     | 学习率                                 |
| `--gamma`          | float     | 0.99     | 折扣因子                               |
| `--buffer_size`    | int       | 100000   | 经验回放缓冲区大小                     |
| `--batch_size`     | int       | 64       | 批量大小                               |
| `--tau`            | float     | 0.005    | 目标网络软更新系数                     |
| `--hidden_dim`     | int       | 64       | 隐藏层维度                             |
| `--env_config`     | str       | None     | 环境配置文件路径（JSON格式）           |
| `--save_dir`       | str       | logs     | 模型和日志保存目录                     |
| `--visualize`      | flag      | False    | 是否保存训练过程视频                   |
| `--video_interval` | int       | 100      | 保存视频的间隔（回合数）               |
| `--video_fps`      | int       | 30       | 视频帧率                               |
| `--video_quality`  | int       | 95       | 视频质量（0-100）                      |
| `--render_width`   | int       | 1024     | 渲染宽度                               |
| `--render_height`  | int       | 768      | 渲染高度                               |

### 测试命令参数 (`test.py`)

| 参数              | 类型      | 默认值   | 说明                                    |
|-------------------|-----------|----------|----------------------------------------|
| `--model_path`    | str       | 必需     | 模型文件路径                           |
| `--num_episodes`  | int       | 5        | 测试的回合数                           |
| `--fps`           | int       | 30       | 视频帧率                               |
| `--video_quality` | int       | 95       | 视频质量（0-100）                      |
| `--render_width`  | int       | 1024     | 渲染宽度                               |
| `--render_height` | int       | 768      | 渲染高度                               |
| `--env_config`    | str       | None     | 环境配置文件路径（JSON格式）           |
| `--save_dir`      | str       | logs     | 测试结果保存目录                       |
| `--heat_map`      | flag      | False    | 是否生成热图                           |
| `--disable_video` | flag      | False    | 禁用视频录制                           |
| `--full_analysis` | flag      | False    | 进行完整分析（更多图表）               |
| `--custom_render` | flag      | False    | 使用自定义渲染设置                     |

## 使用示例

### 训练模型

默认参数训练:
```bash
python src/train.py
```

自定义参数训练:
```bash
python src/train.py --num_episodes 2000 --max_steps 1000 --learning_rate 1e-4 --save_dir logs/custom_training --visualize --video_interval 50
```

使用更大隐藏层和不同折扣因子:
```bash
python src/train.py --hidden_dim 128 --gamma 0.95 --batch_size 128
```

自定义渲染分辨率和录制高质量视频:
```bash
python src/train.py --render_width 1280 --render_height 960 --visualize --video_fps 60 --video_quality 98
```

从已有模型继续训练:
```bash
python src/train.py --model_path logs/20230515_123045/model_best.pt --num_episodes 500
```

### 测试模型

基本测试:
```bash
python src/test.py --model_path logs/20230515_123045/model_best.pt
```

生成热图和完整分析:
```bash
python src/test.py --model_path logs/20230515_123045/model_best.pt --num_episodes 10 --heat_map --full_analysis
```

高分辨率测试视频:
```bash
python src/test.py --model_path logs/20230515_123045/model_best.pt --render_width 1920 --render_height 1080 --fps 60 --video_quality 99
```

自定义环境配置进行测试:
```bash
python src/test.py --model_path logs/20230515_123045/model_best.pt --env_config custom_config.json
```

## 环境配置说明

环境配置定义在 `src/config/environment_config.py` 文件中，包含以下主要参数组：

### 环境基本参数 (ENV_CONFIG)

- **尺寸和小车参数**：环境尺寸、小车尺寸
- **物理参数**：最大/最小速度、加速度、摩擦力、转向参数
- **感知系统**：雷达射线数量和长度
- **初始位置**：小车的起始位置和朝向

### 障碍物参数 (OBSTACLE_CONFIG)

- **静态障碍物**：数量、最小/最大半径
- **动态障碍物**：数量、半径范围、速度范围、方向改变概率

### 目标参数 (TARGET_CONFIG)

- **基本参数**：半径、初始位置
- **移动特性**：移动概率、移动速度、平滑因子

### 奖励参数 (REWARD_CONFIG)

- **基础奖励**：目标达成奖励、步数惩罚
- **行为奖励**：靠近目标奖励、保持速度奖励、位移奖励
- **惩罚**：碰撞惩罚、边界惩罚、原地旋转惩罚、停滞不前惩罚

### 渲染参数 (RENDER_CONFIG)

- **尺寸**：视觉输入尺寸、显示尺寸
- **颜色**：各种元素的颜色设置
- **显示选项**：轨迹、网格等

## 自定义环境配置

可以通过JSON文件提供自定义环境配置，例如：

```json
{
  "width": 1200,
  "height": 900,
  "static_obstacles": {
    "count": 20
  },
  "dynamic_obstacles": {
    "count": 10
  },
  "target": {
    "move_probability": 0.05
  }
}
```

然后使用 `--env_config` 参数指定配置文件：

```bash
python src/train.py --env_config custom_config.json
```

## 可视化输出说明

训练和测试过程会生成多种可视化输出：

1. **训练/测试视频**：显示智能体在环境中的表现，包含信息面板
2. **奖励曲线**：展示每个回合的奖励变化
3. **热图**：显示小车在环境中的活动区域密度
4. **行为分析图**：包括速度、转向、距离等指标的变化趋势
5. **统计图表**：成功率、碰撞率、步数分布等

## 模型文件

训练过程会保存以下模型文件：

- **model_latest.pt**：最新的模型状态
- **model_best.pt**：验证表现最好的模型
- **model_checkpoint_X.pt**：特定步数的检查点模型

模型文件包含：
- Actor-Critic网络的状态字典
- 优化器状态
- 噪声参数 