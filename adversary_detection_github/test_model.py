import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.CNN import AudioClassificationModel
from data.dataloader import AudioDataset
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import warnings


warnings.filterwarnings('ignore')
# Please use your own model path
processor = Wav2Vec2Processor.from_pretrained('/data1/hyh/adversary_detection/utils_models/wav2vec2-base-960h/')
wav2model = Wav2Vec2Model.from_pretrained('/data1/hyh/adversary_detection/utils_models/wav2vec2-base-960h/')


def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def standardize_matrix(matrix):
    mean_val = np.mean(matrix)
    std_val = np.std(matrix)
    standardized_matrix = (matrix - mean_val) / std_val
    return standardized_matrix



def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    attention_weights_list = []
    cost_time = []

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            timex = time.time()
            outputs, attention_weights = model(inputs)
            timey = time.time() - timex
            cost_time.append(timey)
            attention_weights = attention_weights.mean(dim=0).cpu().numpy()

            loss = criterion(outputs, labels.long())
            # predicted = (outputs.squeeze() > 0.5).float()
            _, predicted = torch.max(outputs, 1)
            # print(f'Predicted: {predicted}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item() * inputs.size(0)
            # 计算F1 score
            labels_f1 = labels.cpu().numpy()
            predicted_f1 = predicted.cpu().numpy()
            f1 += f1_score(labels_f1, predicted_f1)
            precision += precision_score(labels_f1, predicted_f1)
            recall += recall_score(labels_f1, predicted_f1)

            attention_weights_list.append(attention_weights)
    print(f'Average Time: {np.mean(cost_time)}')
    return (val_loss / len(val_dataloader), correct / total, f1 / len(val_dataloader), precision / len(val_dataloader),
            recall / len(val_dataloader), attention_weights_list)


def count_label(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    labels = data['labels']
    labels = np.array(labels)
    return len(labels)


device = 'cuda:1'
# 从best_model.pth加载模型
model = AudioClassificationModel(n_fft=512, attention_dim=64,
                                 num_classes=2, wav2vec_model=wav2model,
                                 wav2vec_processor=processor, device=device)


# 加载ces数据集
with open(f'data/json_file/WB/data_test_WB.json', 'r') as f:
    data = json.load(f)
test_audio_paths = data['audio_paths']
test_labels = data['labels']

mask = None
test_dataset = AudioDataset(test_audio_paths, test_labels, mask)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model.load_state_dict(torch.load('best_model/WB/best_model.pth'))
criteria = nn.CrossEntropyLoss()
device = torch.device('cuda:0')
model.to(device)
time1 = time.time()
loss, acc, f1, precision, recall, attention_weights_list = evaluate(model, test_dataloader, criteria, device)
time2 = time.time() - time1
print(f'********** Result **********')
print(f'Loss: {loss}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}')
Nums = count_label(f'data/json_file/WB/data_test_WB.json')
print(f'Time: {time2}\n Average Time: {time2/Nums}')





