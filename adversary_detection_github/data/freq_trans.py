import soundfile
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import librosa.feature
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import torchaudio
import os
import scipy.fftpack
import pywt


processor = Wav2Vec2Processor.from_pretrained('/mnt/hyh/utils_models/wav2vec2-base-960h/')
model = Wav2Vec2Model.from_pretrained('/mnt/hyh/utils_models/wav2vec2-base-960h/')
print('model loaded')


def pad_or_truncate(audio, max_length):
    if len(audio) > max_length:
        return audio[:max_length]
    else:
        return np.pad(audio, (0, max_length - len(audio)), mode='constant')


def extract_features(audio_file_path, n_fft=512, hop_length=128):
    # features = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    # features = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # features = librosa.amplitude_to_db(features, ref=np.max)  # 对数变换
    # features = (features - features.mean()) / features.std()  # 标准化

    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = pad_or_truncate(waveform[0].numpy(), 16000 * 3)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values

    with torch.no_grad():
        features = model(input_values).last_hidden_state
    return features.squeeze(0).numpy()  # 返回 [time_steps, feature_dim] 维度的特征


def apply_dft(features):
    features_fft = np.fft.fft(features, axis=0)
    return np.abs(features_fft)


def apply_dwt(features):
    cA_list, cD_list = [], []
    for i in range(features.shape[1]):
        cA, cD = pywt.dwt(features[:, i], 'db1')
        cA_list.append(cA)
        cD_list.append(cD)
    return np.array(cA_list).T, np.array(cD_list).T


audio_path = '/mnt/hyh/source_3_speaker/302/302-123523-0038.wav'
audio, sr = soundfile.read(audio_path)
wavelet = 'db4'
level = 4
coeffs = pywt.wavedec(audio, wavelet, mode='symmetric', level=level)  # level 表示分解的层数

# coeffs 是一个元组，包含了各个频率带的系数
# (approximations, list_of_details) = coeffs
# approximations 是近似系数，对应于最粗略的频率表示
# list_of_details 是细节系数列表，包含了不同方向和频率的系数

# 打印每个频率带的大小
for i, c in enumerate(coeffs):
    print(f"Level {i+1} coefficients shape: {c.shape}")

coeffs_combined = np.concatenate([c for c in coeffs], axis=-1)
print(f"Combined coefficients shape: {coeffs_combined.shape}")
