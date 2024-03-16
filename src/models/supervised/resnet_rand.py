import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from torchvision.models.segmentation import fcn_resnet101

class ResNetWithRandomForest(pl.LightningModule):
    def __init__(self,  in_channels, out_channels,
                 learning_rate=1e-3, model_params: dict = {}):
        
        super(ResNetWithRandomForest, self).__init__()

        # Load a pre-trained ResNet model
        # self.resnet = fcn_resnet101(num_classes=out_channels,pr)
        
        self.resnet.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7),stride=(2,2), padding=(3,3), bias=False) 
        self.resnet.classifier[4] =nn.Conv2d(512, out_channels, kernel_size=(1,1),stride=(1,1)) 

        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze the parameters of the ResNet model
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define the Random Forest classifier
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # Define input size for ResNet
        self.input_size = in_channels
        # Number of classes
        self.num_classes = out_channels

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
        return features.view(features.size(0), -1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        features = self(x)


        # Train Random Forest classifier on extracted features
        if self.global_step == 0:
            combined_features = features.cpu().numpy()
        else:
            combined_features = np.concatenate((combined_features, features.cpu().numpy()), axis=0)

        # Train the Random Forest classifier on the combined features
        if self.global_step == self.trainer.max_steps - 1:
            self.rf_classifier.fit(combined_features, y.cpu().numpy())
        loss = ...  # Define your loss function
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self(x)
        # Use the trained Random Forest classifier for validation
        y_pred = self.rf_classifier.predict(features.cpu().numpy())
        val_acc = accuracy_score(y.cpu().numpy(), y_pred)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

