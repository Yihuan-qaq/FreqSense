import os
import numpy as np
import librosa
import soundfile as sf
import wave
import matplotlib.pyplot as plt
from tqdm import tqdm


# STFT参数设置
n_fft = 512  # 窗长
hop_length = 128  # 步长
sr = 16000
SHOW = True
FREQ_LOW = [[0, 1000]]
FREQ_HIGH = [[6000, 8000]]
FREQ_BOTH = [[0, 1000], [6000, 8000]]


# 滤波函数群
def filter_global_F(data, threshold):
    data[data < threshold] = 0
    return data


def low_filter_FREQ(ft_matrix, threshold, FREQ):
    n_fft = 512
    start_index = []
    end_index = []
    for i in range(len(FREQ)):
        start_index.append(FREQ[i][0])
        end_index.append(FREQ[i][1])

    # 转换为帧的下标
    temp_start = np.zeros(len(start_index), dtype=int)
    temp_end = np.zeros(len(end_index), dtype=int)
    for i_ in range(len(start_index)):
        temp_start[i_] = int(start_index[i_] // (8000 / (n_fft // 2 + 1)))
        temp_end[i_] = int(end_index[i_] // (8000 / (n_fft // 2 + 1)) + 1)

    for i__ in range(len(start_index)):
        ft_matrix[temp_start[i__]: temp_end[i__], :] = 0
    return ft_matrix


def show_spec(data_maxtrix):
    D = librosa.amplitude_to_db(data_maxtrix, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.colorbar()
    # plt.ylim(0, 8000)
    plt.show()


def show_spec_v2(path):
    f = wave.open(path, 'rb')
    # 得到语音参数
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 得到的数据是字符串，需要将其转成int型
    strData = f.readframes(nframes)
    wavaData = np.frombuffer(strData, dtype=np.int16)
    # 归一化
    # wavaData = wavaData * 1.0 / max(abs(wavaData))
    # .T 表示转置
    wavaData = np.reshape(wavaData, [nframes, nchannels]).T
    f.close()
    # 绘制频谱
    plt.figure(figsize=(20, 5))
    plt.specgram(wavaData[0], Fs=framerate, scale_by_freq=True, sides='default')
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.show()


def mask_frequencies(stft_matrix, low_freq_band=(0, 1000), high_freq_band=(6000, 8000)):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    low_freq_band = [0, 1000]
    high_freq_band = [6000, 8000]

    low_mask = (freqs >= low_freq_band[0]) & (freqs <= low_freq_band[1])
    high_mask = (freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])

    stft_low_masked = np.copy(stft_matrix)
    stft_high_masked = np.copy(stft_matrix)
    stft_both_masked = np.copy(stft_matrix)

    stft_low_masked[low_mask, :] = 0
    stft_high_masked[high_mask, :] = 0
    stft_both_masked[low_mask | high_mask, :] = 0

    return stft_low_masked, stft_high_masked, stft_both_masked


def process_audio_files(input_folder, output_folder_low, output_folder_high, output_folder_both):
    if not os.path.exists(output_folder_low):
        os.makedirs(output_folder_low)
    if not os.path.exists(output_folder_high):
        os.makedirs(output_folder_high)
    if not os.path.exists(output_folder_both):
        os.makedirs(output_folder_both)

    for speaker in tqdm(os.listdir(input_folder), desc='Processing : '):
        speaker_folder = os.path.join(input_folder, speaker)
        if not os.path.exists(os.path.join(output_folder_low, speaker)):
            os.makedirs(os.path.join(output_folder_low, speaker))
        if not os.path.exists(os.path.join(output_folder_high, speaker)):
            os.makedirs(os.path.join(output_folder_high, speaker))
        if not os.path.exists(os.path.join(output_folder_both, speaker)):
            os.makedirs(os.path.join(output_folder_both, speaker))

        for file_name in os.listdir(speaker_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(speaker_folder, file_name)

                # 读取音频文件
                y, _ = sf.read(file_path)

                # 进行STFT
                stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                stft_matrix = np.abs(stft_matrix)
                stft_matrix_low = np.copy(stft_matrix)
                stft_matrix_high = np.copy(stft_matrix)
                stft_matrix_both = np.copy(stft_matrix)

                # 频段屏蔽
                # stft_low_masked, stft_high_masked, stft_both_masked = mask_frequencies(
                #     stft_matrix, sr
                # )
                stft_low_masked = low_filter_FREQ(stft_matrix_low, 100, FREQ_LOW)
                stft_high_masked = low_filter_FREQ(stft_matrix_high, 100, FREQ_HIGH)
                stft_both_masked = low_filter_FREQ(stft_matrix_both, 100, FREQ_BOTH)

                show_spec(stft_matrix)
                show_spec(stft_low_masked)
                show_spec(stft_high_masked)
                show_spec(stft_both_masked)


                # 逆STFT变换
                y_low_masked = librosa.istft(stft_low_masked, hop_length=hop_length)
                y_high_masked = librosa.istft(stft_high_masked, hop_length=hop_length)
                y_both_masked = librosa.istft(stft_both_masked, hop_length=hop_length)


                # 保存屏蔽后的音频
                # sf.write(os.path.join(output_folder_low, speaker, file_name), y_low_masked, sr)
                # sf.write(os.path.join(output_folder_high, speaker, file_name), y_high_masked, sr)
                # sf.write(os.path.join(output_folder_both, speaker, file_name), y_both_masked, sr)

                # global SHOW
                # if SHOW is True:
                #     show_spec_v2(file_path)
                #     show_spec_v2(os.path.join(output_folder_low, speaker, file_name))
                #     show_spec_v2(os.path.join(output_folder_high, speaker, file_name))
                #     show_spec_v2(os.path.join(output_folder_both, speaker, file_name))
                #     SHOW = False


# 查看文件夹下的音频数量
def count_audio(dir_path):
    count = 0
    root, dirs, files = next(os.walk(dir_path))
    # 双层文件夹
    for dir in dirs:
        root_2, dirs_2, files = next(os.walk(os.path.join(root, dir)))
        for file in files:
            if file.find('.wav') != -1:
                count += 1
    print(f'Found {count} audio files')


attack_methods = ['PGD', 'CW2', 'source']
input_folder_dir = "/data1/hyh/adversary_detection_dataset/adv_audio/VCTK/"  # 原始数据集路径
output_folder_low_dir = "/data1/hyh/adversary_detection_dataset/adv_audio_low/"  # 屏蔽低频后的数据集路径
output_folder_high_dir = "/data1/hyh/adversary_detection_dataset/adv_audio_high/"  # 屏蔽高频后的数据集路径
output_folder_both_dir = "/data1/hyh/adversary_detection_dataset/adv_audio_low_high/"  # 屏蔽低频和高频后的数据集路径
for attack_method in attack_methods:
    input_folder = os.path.join(input_folder_dir, attack_method)
    count_audio(input_folder)