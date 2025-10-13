# Environment Configuration

```bash
conda create -n snake python=3.13.4
conda activate snake
# (snake) your_name:snake_ai[master]% 
```

# MLP VS CNN

| 项目       | **MLP**               | **CNN**                      |
| -------- | --------------------- | ---------------------------- |
| 全称       | Multilayer Perceptron | Convolutional Neural Network |
| 连接方式     | 全连接                   | 局部连接（卷积核）                    |
| 参数数量     | 很多                    | 较少（共享权重）                     |
| 主要应用     | 表格数据、简单分类             | 图像、语音、视频                     |
| 是否利用空间结构 | ❌ 否                   | ✅ 是                          |
| 特点       | 通用但不高效                | 擅长图像识别、特征提取                  |
