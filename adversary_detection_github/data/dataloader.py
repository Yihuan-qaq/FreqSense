import soundfile
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import librosa.feature

from torchaudio.transforms import FrequencyMasking, TimeMasking
import torchaudio
import os
import scipy.fftpack


class Augment:
    def __init__(self):
        self.masker_net = torch.nn.Sequential(
            FrequencyMasking(freq_mask_param=100),
            # TimeMasking(time_mask_param=80),
        )
    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio

    def time_shift(self, audio, shift_max=2):
        shift = np.random.randint(shift_max)
        augmented_audio = np.roll(audio, shift)
        return augmented_audio

    def time_mask(self, audio, mask_factor=0.2):
        mask = np.random.randint(0, int(mask_factor * len(audio)))
        start = np.random.randint(0, len(audio) - mask)
        augmented_audio = np.concatenate((audio[:start], audio[start + mask:]))
        return augmented_audio

    def frequency_mask(self, audio, mask_factor=0.2):
        mask = np.random.randint(0, int(mask_factor * len(audio)))
        augmented_audio = audio.copy()
        augmented_audio[:mask] = 0
        augmented_audio[-mask:] = 0
        return augmented_audio

    def masker(self, audio):
        spec = librosa.stft(audio, n_fft=512, hop_length=128, win_length=512, window='hann')
        phase = np.angle(spec)
        spec = np.abs(spec)
        spec = torch.from_numpy(spec)
        spec_fm = self.masker_net(spec)
        spec_fm = spec_fm.numpy()
        spec_fm = spec_fm * phase
        audio = librosa.istft(spec_fm, hop_length=128, win_length=512, window='hann')
        return audio

def apply_dft(features):
    features_fft = np.fft.fft(features, axis=0)
    return np.abs(features_fft)


def pad_or_truncate(audio, max_length):
    if len(audio) > max_length:
        return audio[:max_length]
    else:
        return np.pad(audio, (0, max_length - len(audio)), mode='constant')


def extract_features(audio, n_fft=511, hop_length=128):
    # features = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    features = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    features = librosa.amplitude_to_db(np.abs(features), ref=np.max)  # 对数变换
    features = (features - features.mean()) / features.std()  # 标准化

    # waveform, sample_rate = torchaudio.load(audio)
    # sample_rate = 16000
    # waveform = torch.FloatTensor(audio)
    # # waveform = pad_or_truncate(waveform[0].numpy(), 13700 * 3 * 2)
    # input_values = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values
    #
    # with torch.no_grad():
    #     features = model(input_values).last_hidden_state
        # features = apply_dft(features.squeeze(0).numpy())
    return features  # 返回 [time_steps, feature_dim] 维度的特征


def create_frequency_mask(stft_shape, sr=16000, n_fft=512):
    freq_bins = np.fft.rfftfreq(n_fft, 1 / sr)
    mask = np.zeros(stft_shape[0])

    important_ranges = [(300, 3000), (5500, 6000), (7800, 8000)]

    for low, high in important_ranges:
        mask[(freq_bins >= low) & (freq_bins <= high)] = 1

    return mask


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, mask, sr=16000, n_fft=511, hop_length=128, max_length=3):
        self.audio_paths = audio_paths
        self.labels = labels
        self.mask = mask
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length * sr + 100
        self.augment = Augment()

    def __len__(self):
        return len(self.audio_paths)

    def pad_or_truncate(self, audio):
        if len(audio) > self.max_length:
            return audio[:self.max_length]
        else:
            return np.pad(audio, (0, self.max_length - len(audio)), mode='constant')

    def __getitem__(self, idx):
        audio = soundfile.read(self.audio_paths[idx])[0]
        label = self.labels[idx]
        if label == 1:
            x = np.random.rand()  # 0-1之间的随机数
            if np.random.rand() < 0.25:
                audio = self.augment.add_noise(audio)
            elif np.random.rand() < 0.25:
                audio = self.augment.time_shift(audio)
                # pass
            elif np.random.rand() < 0.25:
                audio = self.augment.masker(audio)
            else:
                pass
        audio = self.pad_or_truncate(audio)
        features = extract_features(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        # Wav2vec2
        # features = features.squeeze(0).numpy()
        # mask = self.mask[..., np.newaxis]
        # stft = stft * mask

        # features = features[np.newaxis, ...]  # Add channel dimension
        # mask = self.mask[..., np.newaxis, np.newaxis]
        # mask = self.mask[..., np.newaxis]
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # return audio, torch.tensor(label, dtype=torch.float32)
