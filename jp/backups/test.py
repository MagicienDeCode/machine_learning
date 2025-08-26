import pyopenjtalk
import numpy as np
from scipy.io import wavfile

text = "単語"
x, sr = pyopenjtalk.tts(text, run_marine=True)

# 归一化到 int16
x_int16 = np.int16(x * 32767)
wavfile.write("output.wav", sr, x_int16)