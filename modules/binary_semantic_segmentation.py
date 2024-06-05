import torch
import torch.nn as nn
import pytorch_lightning as pl

class BinarySegmentationModule(pl.LightningModule):
    """
    A PyTorch Lightning module for binary image segmentation, utilizing a given neural network model.
    This module applies binary cross-entropy with logits as the loss function and evaluates model performance
    using the Intersection over Union (IoU) metric.

    Attributes:
        model (torch.nn.Module): The neural network model for segmentation.
        loss_fn (torch.nn.Module): Loss function to measure the difference between the predictions and actual values.
        lr (float): Learning rate for the optimizer.

    Args:
        model (torch.nn.Module): The model to be used for image segmentation.
        lr (float): Initial learning rate for training the model.
    """

    def __init__(self, model, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input data (images).

        Returns:
            torch.Tensor: Model logits.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step to process one batch during training.

        Args:
            batch (dict): A batch from the dataset containing 'image' and 'mask'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        images, masks = batch['image'], batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to process one batch during validation.

        Args:
            batch (dict): A batch from the dataset.
            batch_idx (int): The index of the batch.
        """
        images, masks = batch['image'], batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.sigmoid(logits) > 0.5
        iou = self.compute_iou(preds, masks)
        self.log('val_loss', loss)
        self.log('val_iou', iou)

    def test_step(self, batch, batch_idx):
        """
        Test step to process one batch during testing.

        Args:
            batch (dict): A batch from the dataset.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Test IoU metric.
        """
        images, masks = batch['image'], batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.sigmoid(logits) > 0.5
        iou = self.compute_iou(preds, masks)
        self.log('test_loss', loss)
        self.log('test_iou', iou)
        return {'test_iou': iou}

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def compute_iou(self, preds, target):
        """
        Compute the Intersection over Union (IoU) for predictions.

        Args:
            preds (torch.Tensor): Predicted outputs.
            target (torch.Tensor): Ground truth labels.

        Returns:
            float: Mean IoU score.
        """
        intersection = torch.logical_and(target, preds).sum(dim=(1, 2)).float()
        union = torch.logical_or(target, preds).sum(dim=(1, 2)).float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()
