import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import ssl
import librosa
import librosa.feature
ssl._create_default_https_context = ssl._create_stdlib_context

sys.path.append(r'/data1/hyh/adversary_detection/model/')
from torchvision import models
from attention import SelfAttention

sr = 16000
n_fft = 511


def mask_frequencies(stft_matrix):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    high_freq_band = [1000, 6000]

    high_mask = (freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])

    stft_low_masked = np.copy(stft_matrix)
    stft_low_masked[high_mask, :] = 0

    return stft_low_masked


# Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 512)
        self.transformer = nn.Transformer(d_model=512, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, feature_dim)
        transformer_out = self.transformer(x, x)
        transformer_out = transformer_out.mean(dim=0)
        out = self.fc(transformer_out)
        return out


# ResNet
class ResNetBackbone(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, example_stft):
        super(CNNModel, self).__init__()
        # 定义CNN网络结构，我的输入特征维度是[16,1,257,626]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 32 * 78, 1024)  # 调整全连接层输入大小
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, mask):
        # x = x * mask  # Apply mask
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class FrequencySelfAttention(nn.Module):
    def __init__(self, freq_dim, attention_dim):
        super(FrequencySelfAttention, self).__init__()
        self.query = nn.Linear(freq_dim, attention_dim)
        self.key = nn.Linear(freq_dim, attention_dim)
        self.value = nn.Linear(freq_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, Time-Size, n_fft/2]
        Q = self.query(x)  # [batch_size, Time-Size, attention_dim]
        K = self.key(x)  # [batch_size, Time-Size, attention_dim]
        V = self.value(x)  # [batch_size, Time-Size, attention_dim]

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                    K.size(-1) ** 0.5)  # [batch_size, Time-Size, Time-Size]
        attention_weights = self.softmax(attention_scores)  # [batch_size, Time-Size, Time-Size]

        # Attention output
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, Time-Size, attention_dim]
        return attention_output, attention_weights


class AudioClassificationModel(nn.Module):
    def __init__(self, n_fft, attention_dim, num_classes, wav2vec_model, wav2vec_processor, device):
        super(AudioClassificationModel, self).__init__()
        self.attention_in_channels = 376
        self.attention_out_channels = 376
        self.num_heads = 8
        self.attention_dropout = 0.5
        # self.self_attention = FrequencySelfAttention(self.attention_in_channels, self.attention_out_channels)
        self.self_attention = SelfAttention(self.num_heads, self.attention_in_channels, self.attention_out_channels,
                                            self.attention_dropout)

        self.conv1 = nn.Conv1d(in_channels=self.attention_out_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 46, 512)
        self.fc_add = nn.Linear(512, 256)

        self.resnet = ResNetBackbone(num_classes)

        self.transformer = TransformerModel(376, num_classes)

        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        self.wav2vec_model = wav2vec_model
        self.wav2vec_processor = wav2vec_processor
        self.device = device
        self.sr = 16000

    # 输入是[batch_size, audio_length]，对每一个audio提取wav2vec特征
    def wav2vec_features(self, inputs_data):
        features = []
        for input_data in inputs_data:
            # stft
            input_data = input_data.cpu().numpy()
            x = librosa.stft(input_data, n_fft=511, hop_length=128)
            # mel
            # input_data = input_data.cpu().numpy()
            # x = librosa.feature.melspectrogram(y=input_data, sr=16000, n_fft=512, hop_length=128, n_mels=128)

            features.append(x)
        if isinstance(features[0], np.ndarray):
            features = [torch.from_numpy(np_array) for np_array in features]
        features = torch.stack(features)
        return features

    def forward(self, x):

        x, attention_weights = self.self_attention(x)  # [batch_size, Time-Size, attention_dim]

        # ResNet
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)

        return x, attention_weights
