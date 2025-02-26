import os
import numpy as np
import json
from sklearn.model_selection import train_test_split


# The folder hierarchy of the adversarial example is：adv_dataset/adv_method/speaker/adv_sample.wav
# The folder hierarchy for normal audio samples is：source_dataset/sample.wav
# The json file format of the generated dataset is：{'audio_paths': [path1, path2, ...], 'labels': [label1, label2, ...]}
def create_json(adv_dir, source_dir, method_type, json_save_root_dir):
    adv_label = 1
    source_label = 0
    audio_paths = []
    labels = []
    json_save_dir = os.path.join(json_save_root_dir, method_type)
    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)
    if method_type == 'TR':
        adv_method = ['PGD', 'FGSM']
    elif method_type == 'WB':
        adv_method = ['CW2', 'CWinf']
    elif method_type == 'BB':
        adv_method = ['fakebob', 'SirenAttack']
    elif method_type == 'ALL':
        adv_method = ['PGD', 'FGSM', 'CW2', 'CWinf', 'fakebob', 'SirenAttack']
    elif method_type == 'VTCK':
        adv_method = ['PGD', 'CW2']
    else:
        raise ValueError('Invalid method type')
    for method in adv_method:
        adv_root_dir = os.path.join(adv_dir, method)
        root, dirs, _ = next(os.walk(adv_root_dir))
        for dir in dirs:
            root_2, dirs_2, files = next(os.walk(os.path.join(root, dir)))
            for file in files:
                if file.endswith('.wav'):
                    # random sample
                    if np.random.rand() < 0.5:
                        continue
                    audio_paths.append(os.path.join(root, dir, file))
                    labels.append(adv_label)

    ADV_NUMS = len(audio_paths)
    print(f'Found {ADV_NUMS} adversarial audio files')
    SOURCE_NUMS = 0

    root_dir = source_dir
    root_2, dirs_2, files = next(os.walk(root_dir))
    for file in files:
        if file.endswith('.wav'):
            audio_paths.append(os.path.join(root_2, file))
            labels.append(source_label)
            SOURCE_NUMS += 1
            if SOURCE_NUMS == ADV_NUMS:
                print(f'Now Found {SOURCE_NUMS} source audio files')
                break

    # Divide the dataset into 7:2:1 and save it to a json file
    audio_paths_train, audio_paths_test, labels_train, labels_test = train_test_split(audio_paths, labels, test_size=0.3, random_state=42)
    audio_paths_val, audio_paths_test, labels_val, labels_test = train_test_split(audio_paths_test, labels_test, test_size=0.33, random_state=42)

    with open(os.path.join(json_save_dir, 'data_train_{}.json'.format(method_type)), 'w') as f:
        json.dump({'audio_paths': audio_paths_train, 'labels': labels_train}, f, indent=4)
    with open(os.path.join(json_save_dir, 'data_val_{}.json'.format(method_type)), 'w') as f:
        json.dump({'audio_paths': audio_paths_val, 'labels': labels_val}, f, indent=4)
    with open(os.path.join(json_save_dir, 'data_test_{}.json'.format(method_type)), 'w') as f:
        json.dump({'audio_paths': audio_paths_test, 'labels': labels_test}, f, indent=4)


# Generate a dataset in json format
def create_data(wav_dir, label):
    audio_paths = []
    labels = []
    root, dirs, _ = next(os.walk(wav_dir))
    for dir in dirs:
        root_2, dirs_2, files = next(os.walk(os.path.join(root, dir)))
        for file in files:
            if file.endswith('.wav'):
                audio_paths.append(os.path.join(root, dir, file))
                labels.append(label)

    print(f'Found {len(audio_paths)} audio files')
    # 保存数据集到json文件
    with open('data_source_librispeech.json', 'w') as f:
        json.dump({'audio_paths': audio_paths, 'labels': labels}, f, indent=4)


# Merge two json files with the same format
def merge_json(json1, json2):
    with open(json1, 'r') as f:
        data1 = json.load(f)
    with open(json2, 'r') as f:
        data2 = json.load(f)

    audio_paths = data1['audio_paths'] + data2['audio_paths']
    labels = data1['labels'] + data2['labels']

    with open('data_total.json', 'w') as f:
        json.dump({'audio_paths': audio_paths, 'labels': labels}, f, indent=4)



# Divide the data set into a ratio of 7:2:1
def split_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    audio_paths = data['audio_paths']
    labels = data['labels']

    audio_paths_train, audio_paths_test, labels_train, labels_test = train_test_split(audio_paths, labels, test_size=0.3, random_state=42)
    audio_paths_val, audio_paths_test, labels_val, labels_test = train_test_split(audio_paths_test, labels_test, test_size=0.33, random_state=42)

    with open('data_train.json', 'w') as f:
        json.dump({'audio_paths': audio_paths_train, 'labels': labels_train}, f, indent=4)
    with open('data_val.json', 'w') as f:
        json.dump({'audio_paths': audio_paths_val, 'labels': labels_val}, f, indent=4)
    with open('data_test.json', 'w') as f:
        json.dump({'audio_paths': audio_paths_test, 'labels': labels_test}, f, indent=4)

