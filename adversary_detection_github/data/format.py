import soundfile
import librosa
import numpy as np
import os


# flac转wav
def flac2wav(flac_path):
    data, samplerate = soundfile.read(flac_path)
    wav_path = flac_path.replace('.flac', '.wav')
    soundfile.write(wav_path, data, samplerate)
    # 删除flac文件
    os.remove(flac_path)


# 两层文件夹结构，第一层为类别，第二层为音频文件，实现dir转wav
def dir2wav(dir_path):
    root, dirs, _ = next(os.walk(dir_path))
    for dir in dirs:
        root_2, dirs_2, files = next(os.walk(os.path.join(root, dir)))
        for file in files:
            if file.endswith('.flac'):
                flac2wav(os.path.join(root, dir, file))


# 统计文件夹下的音频数量
def count_audio(dir_path):
    root, dir, files = next(os.walk(dir_path))
    count = 0
    for file in files:
        if file.find('.wav') != -1:
            count += 1
    print(f'Found {count} audio files')


count_audio('/mnt/hyh/librispeech_10000_source/')