import numpy as np
import os, glob, time, copy, random, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import random
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from torchsampler import ImbalancedDatasetSampler

from torchmetrics import Accuracy, F1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
import json



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

torch.set_float32_matmul_precision('highest')

# Data Augmentation
class ImageTransform():
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        res = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.data_transform = {
            'train': res,
            'val': val,
            'test': val
        }
        
    def __call__(self, img, phase):
        img = img.convert(mode='RGB')
        return self.data_transform[phase](img)

def get_label(path):
    return 1 if "/cat/" in path else 0

class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None, phase='train'):    
        self.images = images
        self.raw_labels = labels
        self.labels = [
            np.array([
                l,
                abs(1-l),
            ])
            for l in labels
        ]
        self.transform = transform or ImageTransform()
        self.phase = phase

        assert len(images) == len(labels)
        
    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return self.raw_labels
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        # Transformimg Image
        img_transformed = self.transform(img, self.phase)
    
        return img_transformed, torch.from_numpy(self.labels[idx]).float()

batch_size = 64
num_workers = 16


class MyModel(L.LightningModule):
    def __init__(self, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,3), nn.ReLU(), nn.MaxPool2d(2,2)) 

        self.conv_alt1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(),  nn.AvgPool2d(2,2)) 
        self.conv_alt2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.AvgPool2d(2,2)) 
        self.conv_alt3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.AvgPool2d(2,2)) 
        self.conv_alt4 = nn.Sequential(nn.Conv2d(64,128,3), nn.ReLU(), nn.AvgPool2d(2,2)) 

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential( 
            nn.Dropout(p=.1),
            nn.Linear(36864, 512), 
            nn.ReLU(), 

            nn.Dropout(p=.1),
            nn.Linear(512, 256), 
            nn.ReLU(), 

            nn.Linear(256, 128), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Linear(128,2))
        self.criterion = nn.BCELoss()
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        x = x.to(device)

        q = torch.clone(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        
        q = self.conv_alt1(q)
        q = self.conv_alt2(q)
        q = self.conv_alt3(q)
        q = self.conv_alt4(q)
        q = self.flatten(q)

        x = torch.cat((x, q), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x) 
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-8)
        return optimizer
    
    def train_dataloader(self):
        # transforms
        return torch.utils.data.DataLoader(
            dataset = train_ds,
            batch_size = batch_size,
            #shuffle=True,
            num_workers=num_workers,
            sampler=ImbalancedDatasetSampler(train_ds),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = test_ds, 
            sampler=ImbalancedDatasetSampler(test_ds),
            batch_size = batch_size,
            #shuffle=True,
            num_workers=num_workers
        )
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log('tr_loss', loss, prog_bar=True)

        self.accuracy(pred, y)
        self.f1(pred, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss, prog_bar=True)

        # f1 should be preferred over accurracy as dataset is imbalanced now
        self.f1(pred, y)
        self.log('val_f1', self.f1, prog_bar=True)

        self.accuracy(pred, y)
        self.log('val_acc', self.accuracy, prog_bar=True)
        return loss

def main():
    model = MyModel.load_from_checkpoint("models_ok/sample-cat-or-not-v0_0_1-epoch=07-val_loss=0.13-val_acc=0.95-val_f1=0.95.ckpt")
    model.batch_size = batch_size

    trainer = L.Trainer(
        accelerator=device,
        max_epochs=10000, 
        callbacks=[
            ModelCheckpoint(
                monitor='val_f1',
                dirpath='models/',
                filename='sample-cat-or-not-v0_0_1-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.2f}',
                save_top_k=3,
                mode='max'
            )
        ]
    )

    # Train the model ⚡
    trainer.fit(model)