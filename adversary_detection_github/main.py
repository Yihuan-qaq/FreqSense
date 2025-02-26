import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model.CNN import AudioClassificationModel
from data.dataloader import AudioDataset, create_frequency_mask, extract_features
import wandb
import json
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from hydra import initialize, compose
from omegaconf import DictConfig
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

# Please use your own model path
processor = Wav2Vec2Processor.from_pretrained('/data1/hyh/adversary_detection/utils_models/wav2vec2-base-960h/')
wav2model = Wav2Vec2Model.from_pretrained('/data1/hyh/adversary_detection/utils_models/wav2vec2-base-960h/')
print('model loaded')


def create_prior_weights(n_fft=512, sr=16000):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    prior_weights = np.ones_like(freqs)
    prior_weights[(freqs >= 300) & (freqs <= 3000)] = 2.0
    prior_weights[(freqs >= 5500) & (freqs <= 6000)] = 2.0
    prior_weights[(freqs >= 7800) & (freqs <= 8000)] = 2.0
    return torch.tensor(prior_weights, dtype=torch.float32)


def train(model, train_dataloader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        t0 = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(train_dataloader)
    return epoch_loss


def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    f1 = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            t0 = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels.long())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item() * inputs.size(0)
            labels_f1 = labels.cpu().numpy()
            predicted_f1 = predicted.cpu().numpy()
            f1 += f1_score(labels_f1, predicted_f1)
            # print(f'one batch_size time: {time.time() - t0}')
    return val_loss / len(val_dataloader), correct / total, f1 / len(val_dataloader)


def main(cfg: DictConfig):
    wandb.init(project=cfg.project.name,
               name='{}_{}'.format(time.strftime(cfg.project.time_format), cfg.project.model_name))
    save_dir = os.path.join(cfg.model.save_dir, cfg.project.method)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
        print('Create save dir:', save_dir)
    model_save_path = os.path.join(cfg.model.save_dir, cfg.project.method, 'best_model.pth')

    with open(cfg.data.train_json, 'r') as f:
        data = json.load(f)
    train_audio_paths = data['audio_paths']
    train_labels = data['labels']

    with open(cfg.data.val_json, 'r') as f:
        data = json.load(f)
    val_audio_paths = data['audio_paths']
    val_labels = data['labels']

    mask = None
    device = torch.device(cfg.train.device)
    train_dataset = AudioDataset(train_audio_paths, train_labels, mask)
    val_dataset = AudioDataset(val_audio_paths, val_labels, mask)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    prior_weights = create_prior_weights(cfg.train.n_fft, cfg.train.sr)
    model = AudioClassificationModel(n_fft=cfg.train.n_fft, attention_dim=cfg.train.attention_dim,
                                     num_classes=cfg.train.num_classes, wav2vec_model=wav2model, wav2vec_processor=processor, device=device)

    model.to(device)
    weights = torch.tensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

    best_acc = cfg.train.best_acc
    for epoch in tqdm(range(cfg.train.num_epochs)):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, scheduler)
        test_loss, test_accuracy, f1 = evaluate(model, val_dataloader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{cfg.train.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, ACC: {test_accuracy:.4f}, F1: {f1:.4f}")
        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_accuracy, "test_f1": f1})
        if test_accuracy > best_acc:
            best_acc = test_accuracy

            torch.save(model.state_dict(), model_save_path)
            print('Model saved!', 'Best accuracy:', best_acc)
        # Save the last five rounds of models
        if epoch >= cfg.train.num_epochs - 5:
            torch.save(model.state_dict(), os.path.join(cfg.model.save_dir, cfg.project.method, f'model_{epoch}.pth'))

    wandb.finish()


if __name__ == "__main__":
    with initialize(config_path="config"):
        cfg = compose(config_name="config_WB")
        main(cfg)
