import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer


class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """

    def __init__(self, model_type, in_channels, out_channels,
                 learning_rate=1e-3, model_params: dict = {}):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model

        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        if model_type == "UNet":
            self.model = UNet(**model_params)
        elif model_type == "SegmentationCNN":
            self.model = SegmentationCNN(**model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)

        # define performance metrics for segmentation task
        # such as accuracy per class accuracy, average IoU, per class IoU,
        # per class AUC, average AUC, per class F1 score, average F1 score
        # these metrics will be logged to weights and biases

        # not sure if iou is correct

        self.acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=out_channels)
        # self.iou = torchmetrics.detection.IntersectionOverUnion(num_classes=out_channels, reduction='none')
        self.f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=out_channels, average='none')
        self.auroc = torchmetrics.classification.AUROC(task='multiclass', num_classes=out_channels, average='none')

        # self.avg_IoU = torchmetrics.detection.IntersectionOverUnion(num_classes=out_channels, reduction='macro')
        self.avg_AUC = torchmetrics.classification.AUROC(task='multiclass', num_classes=out_channels, average='macro')
        self.avg_F1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=out_channels, average='macro')

    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        return self.model.forward(X)

    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height), 
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, mask, _ = batch
        sat_img, mask = sat_img.float(), mask.squeeze(1).long()

        logits = self.forward(sat_img)
        loss = nn.functional.cross_entropy(logits, mask)
        self.log('train_loss', loss)

        return loss


    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the 

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, mask, _ = batch
        sat_img, mask = sat_img.float(), mask.squeeze(1).long()

        logits = self.forward(sat_img)
        loss = nn.functional.cross_entropy(logits, mask)
        self.log('val_loss', loss)

        self.log('accuracy per class', self.acc(logits, mask))
        self.log('F-1 per class', self.f1(logits, mask))
        self.log('AUROC per class', self.auroc(logits, mask))
        self.log('average AUROC', self.avg_AUC(logits, mask))
        self.log('average F-1', self.avg_F1(logits, mask))

        return loss

    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
