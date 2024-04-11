import numpy as np

def generate_audio(duration, sample_rate = 44100, frequency = 440, deleash = 1):
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * time) / deleash 
    audio_data = (audio_data * 32767).astype(np.int16)  # 将数据缩放为16位整数
    return audio_data.tobytes()