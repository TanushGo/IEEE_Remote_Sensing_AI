import numpy as np
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, jaccard_score
from torch.utils.data import DataLoader


class RandomForestsClassifier(pl.LightningModule):
    def __init__(
        self, n_estimators=100, max_depth=None, num_features=None, num_classes=None
    ):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
        self.num_classes = num_classes

    def prepare_data(self, train_loader: DataLoader, val_loader: DataLoader):
        # This method is intended to prepare data for RF, it's not a standard PL method
        # You should call this method manually in your training script before training starts
        self.train_features, self.train_labels = self._prepare_dataset(train_loader)
        self.val_features, self.val_labels = self._prepare_dataset(val_loader)

    def _prepare_dataset(self, loader: DataLoader):
        features = []
        labels = []

        for batch in loader:
            # Assuming the batch is a tuple of (features, labels)
            X, y = batch
            # Reshape X and convert to numpy
            X = X.view(X.size(0), -1).numpy()
            y = y.numpy()

            features.append(X)
            labels.append(y)

        features = np.vstack(features)
        labels = np.concatenate(labels)

        return features, labels

    def forward(self, X):
        # Random Forest does not use this method for prediction
        # It's defined here for compatibility with PyTorch Lightning
        pass

    def training_step(self, batch, batch_idx):
        # Fit the model to the prepared training data
        self.model.fit(self.train_features, self.train_labels)
        # Since RF does not use mini-batch training, we do not use `batch` and `batch_idx`
        return None

    def validation_step(self, batch, batch_idx):
        # Perform prediction on the validation set
        preds = self.model.predict(self.val_features)
        # Calculate metrics
        acc = accuracy_score(self.val_labels, preds)
        f1 = f1_score(self.val_labels, preds, average="weighted")
        roc_auc = roc_auc_score(
            self.val_labels,
            self.model.predict_proba(self.val_features),
            multi_class="ovr",
            average="weighted",
        )
        iou = jaccard_score(self.val_labels, preds, average="weighted")

        self.log("val_acc", acc)
        self.log("val_f1", f1)
        self.log("val_roc_auc", roc_auc)
        self.log("val_iou", iou)

    def configure_optimizers(self):
        # Random Forest does not use an optimizer
        return None
