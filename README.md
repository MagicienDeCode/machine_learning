# Machine Learning

# Install
```
pip install numpy pandas scikit-learn matplotlib lazypredict seaborn -U
```

```bash
# mac os create python env

$ mkdir -p $HOME/.venvs  # create a folder for all virtual environments 
$ python3 -m venv $HOME/.venvs/p3env  # create p3env

alias activep3="source /Users/xiangli/.venvs/p3env/bin/activate"

```

## 线性模型
- 线性回归 (Linear Regression): 用于回归问题，试图找到输入特征和输出变量之间的线性关系。
- 逻辑回归 (Logistic Regression): 用于分类问题，输出的是一个概率值，表示某个样本属于某个类别的概率。

## 决策树模型
- 决策树 (Decision Tree): 一种非参数化的监督学习方法，可以用于分类和回归。通过对特征的条件判断进行分裂。
- 随机森林 (Random Forest): 集成多棵决策树，通过投票或平均的方式提升模型的准确性和防止过拟合。
- 梯度提升树 (Gradient Boosting Trees): 通过逐步添加新的决策树来纠正前面所有树的误差，常用于分类和回归问题。

## 支持向量机 (SVM)
- 支持向量机 (Support Vector Machine): 用于分类问题，通过寻找最佳分割超平面来最大化类间距离。也有支持向量回归 (SVR) 版本用于回归问题。

## 神经网络和深度学习
- 多层感知器 (MLP): 一种前馈神经网络，包含一个或多个隐藏层。适用于回归和分类问题。
- 卷积神经网络 (CNN): 特别适用于处理图像数据，通过卷积层提取图像特征。
- 循环神经网络 (RNN): 适用于处理序列数据，如时间序列和自然语言处理。长短期记忆网络 (LSTM) 和门控循环单元 (GRU) 是RNN的改进版本。

## 聚类模型
- K均值聚类 (K-Means Clustering): 一种非监督学习方法，通过迭代优化簇中心，最小化簇内距离平方和。
- 层次聚类 (Hierarchical Clustering): 通过构建树状层次结构来进行聚类。

## 降维模型
- 主成分分析 (PCA): 一种线性降维方法，通过提取数据的主要成分来降低数据维度。
- t-SNE (t-Distributed Stochastic Neighbor Embedding): 一种非线性降维方法，特别适用于高维数据的可视化。

## 贝叶斯模型
- 朴素贝叶斯 (Naive Bayes): 基于贝叶斯定理的分类方法，假设特征之间是独立的。

## 集成学习
- Adaboost: 通过组合多个弱分类器来提升分类性能，逐步减少难分类样本的权重。
- XGBoost: 一种高效的梯度提升算法，常用于竞赛和实际应用中。