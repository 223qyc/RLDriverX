# RLDriverX: 强化学习驱动的智能自动驾驶系统

<div align="center">
  <img src="https://img.shields.io/badge/状态-开发中-brightgreen" alt="项目状态">
  <img src="https://img.shields.io/badge/版本-1.0.0-blue" alt="版本">
  <img src="https://img.shields.io/badge/框架-PyTorch-orange" alt="框架">
  <img src="https://img.shields.io/badge/领域-强化学习-purple" alt="领域">
</div>

## 项目概述

RLDriverX 是一个基于强化学习的渐进式二维自动驾驶系统，通过三个发展阶段展示了智能体从简单环境到复杂场景的学习过程。本项目实现了自动驾驶算法，并建立了完整的仿真环境和评估系统。

## 项目结构

项目分为三个主要阶段：

1. **First Try-简单场景下智能体的诞生**：实现基础导航能力
2. **Second Try-随机目标且优化算法的智能体**：优化算法和随机目标
3. **Third Try-复杂场景实现与多模态集成**：多模态感知与复杂场景

## 第一阶段：简单场景下智能体的诞生

### 核心实现
- 800×800像素的二维网格世界
- 8个方向的距离传感器
- 基础DQN算法
- 简单的奖励机制

### 技术细节
- 环境尺寸：800×800
- 传感器数量：8个
- 传感器范围：25单位
- 障碍物数量：40个
- 动作空间：前进、左转、右转

### 实验结果
- 基础避障能力
- 简单的路径规划

#### 可视化结果

1. 训练过程分析：
![训练奖励曲线](First%20Try-简单场景下智能体的诞生/training_rewards.png)
*图1：训练过程中的奖励变化曲线*

2. 策略分析：
![策略地图](First%20Try-简单场景下智能体的诞生/visualizations/policy_map.png)
*图2：智能体在不同状态下的策略分布*

3. Q值分布：
![Q值分布](First%20Try-简单场景下智能体的诞生/visualizations/q_values_3d.png)
*图3：Q值的三维分布图*

4. 训练统计：
![训练统计](First%20Try-简单场景下智能体的诞生/visualizations/episode_stats.png)
*图4：训练过程中的各项统计指标*

#### 演示视频
- [基础评估演示](First%20Try-简单场景下智能体的诞生/videos/evaluation.mp4)：展示智能体在基础场景中的表现
- [增强评估演示](First%20Try-简单场景下智能体的诞生/videos/evaluation_enhanced.mp4)：展示智能体在增强场景中的表现
- [复杂场景演示](First%20Try-简单场景下智能体的诞生/videos/evaluation_complex.mp4)：展示智能体在复杂场景中的表现
- [样本回合演示](First%20Try-简单场景下智能体的诞生/videos/sample_episode.mp4)：展示单个训练回合的完整过程

## 第二阶段：随机目标且优化算法的智能体

### 核心改进
- 随机位置目标
- 障碍物数量增加到60个
- 传感器范围扩展到50单位
- 传感器数量增加到16个
- 双重DQN和优先经验回放

### 技术亮点
- 优先经验回放实现
- 双重DQN架构
- 改进的网络结构
- 详细的超参数优化

### 实验结果
- 成功率：提升至85%以上
- 平均步数：减少约40%
- 更好的泛化能力
- 更稳定的训练过程

#### 训练效果
![训练指标](Second%20Try-随机目标且优化算法的智能体/training_metrics.png)
*图5：训练过程中的各项指标变化*

#### 演示视频
[点击查看评估演示视频](Second%20Try-随机目标且优化算法的智能体/videos/evaluation.mp4)

视频展示了智能体在以下场景中的表现：
- 随机目标位置导航
- 复杂障碍物环境中的避障
- 平滑的转向和速度控制
- 稳定的路径规划

## 第三阶段：复杂场景实现与多模态集成

### 系统架构
- 模块化设计
- 多模态感知系统
- 课程学习机制
- 完整的评估系统

### 核心功能
- 视觉输入处理
- 激光雷达模拟
- 向量状态数据
- 多模态融合网络

### 实验结果
- 多模态感知能力
- 复杂场景适应
- 动态障碍物处理
- 移动目标追踪

#### 训练过程分析

1. 训练过程指标图：
![训练过程指标图](Third%20Try-复杂场景实现与多模态集成/logs/20250519_113211/final_plots/combined_metrics.png)
*图6：训练过程中的综合指标变化*

2. 损失函数变化：
![损失函数变化](Third%20Try-复杂场景实现与多模态集成/logs/20250519_113211/final_plots/losses.png)
*图7：训练过程中的损失函数变化*

3. 距离-速度-转向分析：
![距离-速度-转向分析](Third%20Try-复杂场景实现与多模态集成/logs/20250519_113211/final_plots/distance_speed_rotation.png)
*图8：智能体的距离、速度和转向角变化*

4. 碰撞与成功率统计：
![碰撞与成功率统计](Third%20Try-复杂场景实现与多模态集成/logs/20250519_113211/final_plots/collision_success.png)
*图9：训练过程中的碰撞次数和成功率统计*

5. 活动区域热图：
![活动区域热图](Third%20Try-复杂场景实现与多模态集成/logs/20250519_113211/final_heatmap.png)
*图10：智能体在环境中的活动区域分布*

## 使用指南

### 环境要求
- Python 3.7+
- PyTorch 1.7.0+
- NumPy 1.19.2+
- Matplotlib 3.3.2+
- Pygame 2.0.0+
- OpenCV 4.4.0+

### 安装步骤
1. 克隆项目：
```bash
git clone https://github.com/223qyc/RLDriverX.git
cd RLDriverX
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 运行说明

1. 第一阶段（基础DQN）：
```bash
cd "First Try-简单场景下智能体的诞生"
python main.py
```

2. 第二阶段（优化算法）：
```bash
cd "Second Try-随机目标且优化算法的智能体"
python main.py
```

3. 第三阶段（多模态集成）：
```bash
cd "Third Try-复杂场景实现与多模态集成"
python train.py
```

## 项目特点
- 渐进式学习框架
- 模块化设计
- 完整的评估系统
- 丰富的可视化功能
- 详细的文档说明

## 注意事项
- 第三阶段十分粗糙，还有很大的改进余地
- 第三阶段由于设备限制，实际展示是采用的最简化训练与检测，所以效果很差情理之中
- 在第三阶段使用AI协作较多，这里必须指出

## 许可证
本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

