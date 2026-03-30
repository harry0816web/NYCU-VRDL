import os
import random
import wandb
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torch.nn.functional as nn_F
from torchvision.models import (
    ResNeXt50_32X4D_Weights,
    ResNet50_Weights,
    resnet50,
    resnext50_32x4d,
)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.datasets import ImageFolder
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# ==== 0. set environment variables ====
# TODO: replace with your own wandb api key
os.environ['WANDB_API_KEY'] = '1234567890'
# base dir
base_dir = './'

# ==== 1. Dataloader with Augmentation ====


class RandomShortestSize:
    def __init__(self, min_size=256, max_size=480):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        # 隨機決定一個目標短邊長度
        target_size = random.randint(self.min_size, self.max_size)
        return F.resize(img, target_size, antialias=True)


# train set
train_transform = transforms.Compose([
    RandomShortestSize(min_size=256, max_size=480),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # add random erase
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

# val set
val_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ==== 2. Dataset Setup ====
train_dataset = ImageFolder(
    root=base_dir + 'data/train',
    transform=train_transform)
val_dataset = ImageFolder(root=base_dir + 'data/val', transform=val_transform)
batch_size_cfg = 64

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size_cfg,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Total training images: {len(train_dataset)}")
print(f"Image tensor shape: {train_dataset[0][0].size()}")
print(f"Total validation images: {len(val_dataset)}")

# set seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# config
config_defaults = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "learning_rate_backbone": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "batch_size": batch_size_cfg,
    "epochs": 100,
    "architecture": "ResNext-50-Pretrained-Bagging"
}

seeds = [42, 67, 6767]

# ==== 3. Bagging Training Loop ====
for seed in seeds:
    set_seed(seed)

    # bootstrap sampler
    bootstrap_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset)
    )

    # train loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_cfg,
        sampler=bootstrap_sampler,
        num_workers=2,
        pin_memory=True
    )

    # init model backbone
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    model = resnext50_32x4d(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 100)
    )

    # init wandb
    run = wandb.init(
        entity="harry940816-national-yang-ming-chiao-tung-university",
        project="DLCV_hw1",
        config=config_defaults,
        name=f"resnext_bagging_seed_{seed}",
        reinit=True
    )
    cfg = wandb.config

    # set device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # separate params for backbone and fc
    fc_params = list(model.fc.parameters())
    backbone_params = [
        p for name,
        p in model.named_parameters() if "fc" not in name]

    # add label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg.learning_rate * 0.1},
        {'params': fc_params, 'lr': cfg.learning_rate}
    ], weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6
    )

    # set training loop parameters
    global_step = 0
    best_val_loss = float('inf')
    result_dir = os.path.join(base_dir, 'resnext_ensemble_models')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f'best_model_{seed}.pth')
    epochs_without_improvement = 0

    print("Start training Bagging model (Seed: {seed})!")

    # training loop
    for epoch in range(cfg.epochs):
        # training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            global_step += 1

            # print progress every 100 batches
            if (i + 1) % 100 == 0:
                acc = 100 * correct / total
                print(
                    f"Epoch [{epoch + 1}/{cfg.epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Global Step: {global_step}, "
                    f"Loss: {loss.item():.4f}, Train Acc: {acc:.2f}%"
                )

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1} summary:")
        print(
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Acc: {val_acc:.2f}%"
        )

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "train/loss": avg_train_loss,
            "train/accuracy": train_acc,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr
        }, step=epoch)

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
            print("Model saved!")
        else:
            epochs_without_improvement += 1

        scheduler.step()

    wandb.finish()
    print(f"Seed {seed} training complete.")

print("All Bagging models training complete!")


# ==== 4. Testing  ====
class FlatTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted([f for f in os.listdir(
            root) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.images[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]


try:
    idx_to_class = train_dataset.classes
except NameError:
    train_dataset = ImageFolder(
        root=base_dir + 'data/train',
        transform=val_transform)
    idx_to_class = train_dataset.classes

test_dataset = FlatTestDataset(
    root=base_dir + 'data/test',
    transform=val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size_cfg,
    shuffle=False,
    num_workers=2)

# build three same models
model_1 = resnext50_32x4d(weights=None)
model_1.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 100))
model_2 = resnext50_32x4d(weights=None)
model_2.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 100))
model_3 = resnext50_32x4d(weights=None)
model_3.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 100))

# load the weights of three models
model_1.load_state_dict(
    torch.load(
        'resnext_ensemble_models/best_model_42.pth',
        map_location=device))
model_2.load_state_dict(
    torch.load(
        'resnext_ensemble_models/best_model_67.pth',
        map_location=device))
model_3.load_state_dict(
    torch.load(
        'resnext_ensemble_models/best_model_6767.pth',
        map_location=device))

model_1.to(device).eval()
model_2.to(device).eval()
model_3.to(device).eval()

predictions = []
with torch.no_grad():
    for inputs, filenames in tqdm(
            test_loader, desc="Bagging Ensemble Predicting"):
        inputs = inputs.to(device)

        # get the probabilities of three models
        prob_1 = nn_F.softmax(model_1(inputs), dim=1)
        prob_2 = nn_F.softmax(model_2(inputs), dim=1)
        prob_3 = nn_F.softmax(model_3(inputs), dim=1)

        # average the three probabilities (soft voting)
        avg_probs = (prob_1 + prob_2 + prob_3) / 3.0
        _, preds = torch.max(avg_probs, 1)

        for f, p in zip(filenames, preds.cpu().tolist()):
            image_name = os.path.splitext(f)[0]
            pred_label = idx_to_class[p]
            predictions.append((image_name, pred_label))

with open(os.path.join(base_dir, 'prediction.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'pred_label'])
    writer.writerows(predictions)

print(f"預測完成，共 {len(predictions)} 筆，已儲存至 prediction.csv")
