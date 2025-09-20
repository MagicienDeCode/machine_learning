# Machine Learning
* 机器学习（Machine Learning, ML）是人工智能（Artificial Intelligence, AI）的一个分支，它让计算机系统能够通过数据（data）来学习（learn）和改进（improve），而无需被明确地编程。简单来说，我们不是告诉计算机如何解决问题，而是提供大量数据，让它自己找出其中的规律和模式。

## How It Works

1. 数据收集（Data Collection）
2. 特征工程（Feature Engineering）
3. 模型训练（Model Training）
4. 模型评估（Model Evaluation）
5. 部署与优化（Deployment & Optimization）

## Main Types of ML

1. 监督学习（Supervised Learning）: 这是最常见的一类。模型通过**带标签的数据（labeled data）**进行训练，即每条数据都包含输入和对应的正确答案。
    - 应用场景（Applications）：预测房价、垃圾邮件过滤（spam filtering）、图像分类（image classification）。
    - 举例：给模型提供大量照片，并告诉它哪些是猫，哪些是狗。训练后，模型就能对新的照片进行分类（classify）。
2. 无监督学习（Unsupervised Learning）: 模型在**无标签的数据（unlabeled data）**上进行训练，它需要自己找出数据中的结构和模式。
    - 应用场景（Applications）：客户细分（customer segmentation）、异常检测（anomaly detection）、市场篮子分析（market basket analysis）。
    - 举例：给模型提供一群客户的购买数据，它会根据购买习惯将客户**聚类（cluster）**成不同的群体，而事先没有人告诉它要如何分组。
3. 强化学习（Reinforcement Learning）: 模型通过与**环境（environment）的交互（interaction）来学习。它会接收奖励（reward）或惩罚（penalty）**作为反馈，目标是最大化累积奖励。
    - 应用场景（Applications）：自动驾驶、机器人控制、下棋或玩游戏（如AlphaGo）。
    - 训练一个AI玩游戏。当它做出好的动作（比如跳过障碍物）时，就给它一个正向奖励（reward）；当它做出不好的动作时，就给它一个负向惩罚（punishment）。
