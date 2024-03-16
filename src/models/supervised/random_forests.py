import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pytorch_lightning as pl

# Define Random Forest model
class RandomForests(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super(RandomForests, self).__init__()
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        x, y, _= batch
        features = x.view(x.size(0), -1).numpy()
        self.rf_classifier.fit(features, y)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = x.view(x.size(0), -1).numpy()
        y_pred = self.rf_classifier.predict(features)
        val_acc = accuracy_score(y, y_pred)
        self.log('val_acc', val_acc, on_epoch=True)
        return

    def configure_optimizers(self):
        return []
