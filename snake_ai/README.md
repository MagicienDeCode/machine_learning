# Environment Configuration

```bash
# Create a conda environment named snake with Python version 3.8.16
conda create -n snake python=3.8.16
conda activate snake
# (snake) your_name:snake_ai[master]% 

# [Optional] To use GPU for training, manually install the full version of PyTorch
# conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia window
conda install pytorch torchvision torchaudio -c pytorch

# [Optional] Run the script to test if PyTorch can successfully call the GPU
python utils/check_gpu_status.py 
# GPU is not available.

# Install external code libraries
pip install -r requirements.txt

# pip install "pip<24.1" wheel==0.38.4
# pip install setuptools==59.5.0
# brew install cmake swig
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
