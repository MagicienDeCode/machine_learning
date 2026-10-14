import torch
import collections
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.utils.radam import RAdam

# 允许自定义类和常用内置类型加载
torch.serialization.add_safe_globals([
    XttsConfig,
    RAdam,
    collections.defaultdict,
    dict
])

# 初始化 TTS 模型
tts = TTS("tts_models/ja/kokoro/tacotron2-DDC", gpu=False)

# 合成语音到文件
tts.tts_to_file("おはようございます。今日はいい天気ですね。", file_path="voice.wav")