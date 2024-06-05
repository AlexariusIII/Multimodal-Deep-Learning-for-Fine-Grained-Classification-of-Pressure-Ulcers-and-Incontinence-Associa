import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import Accuracy, F1Score, AveragePrecision, AUROC

class ImageClassificationModule(LightningModule):
    """
    A PyTorch Lightning module for multiclass image classification tasks. It incorporates
    metrics such as accuracy, F1 score, area under the ROC curve, and precision-recall AUC.
    
    Attributes:
        model (torch.nn.Module): The neural network model used for classification.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        loss (torch.nn.Module): Loss function, specifically CrossEntropyLoss for multiclass.
    """
    def __init__(self, model, lr, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes, average='macro')
        self.auroc = AUROC(num_classes=num_classes)
        self.prAuc = AveragePrecision(num_classes=num_classes, average='macro')

    def forward(self, image, cat_features):
        """
        Forward pass through the model.
        """
        return self.model(image, cat_features)

    def _shared_step(self, batch, train=False):
        images, true_labels, cat_features = batch['image'], batch['label'], batch['cat_features']
        logits = self(images, cat_features)
        loss = self.loss(logits, true_labels)
        return loss, true_labels, logits

    def training_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch, train=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.accuracy(logits, true_labels), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.accuracy(logits, true_labels), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1(logits, true_labels), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        self.log("test_acc", self.accuracy(logits, true_labels), prog_bar=True)
        self.log("test_f1", self.f1(logits, true_labels), prog_bar=True)
        self.log("test_auroc", self.auroc(logits, true_labels), prog_bar=True)
        self.log("test_prAuc", self.prAuc(logits, true_labels), prog_bar=True)
        return {'test_acc': self.accuracy.compute(), 'test_f1': self.f1.compute()}

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

class ImageClassificationModuleMultiLabel(LightningModule):
    """
    A PyTorch Lightning module for multilabel image classification tasks. It includes metrics
    for evaluating performance on multilabel classification tasks such as accuracy, F1 score,
    area under the ROC curve, and precision-recall AUC.
    
    Attributes are similar to the multiclass version, but loss function and metrics are tailored
    for multilabel tasks.
    """
    def __init__(self, model, lr, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(num_labels=num_classes, average='macro', multiclass=False)
        self.f1 = F1Score(num_labels=num_classes, average='macro', multiclass=False)
        self.auroc = AUROC(num_labels=num_classes, multiclass=False)
        self.prAuc = AveragePrecision(num_labels=num_classes, average='macro', multiclass=False)

    def forward(self, image, cat_features):
        """
        Forward pass through the model.
        """
        return self.model(image, cat_features)

    def _shared_step(self, batch, train=False):
        images, true_labels, cat_features = batch['image'], batch['label'], batch['cat_features']
        logits = self(images, cat_features)
        loss = self.loss(logits, true_labels.float())
        sigmoid_logits = torch.sigmoid(logits)
        predicted_labels = (sigmoid_logits > 0.5).float()
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, train=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.accuracy(predicted_labels, true_labels), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.accuracy(predicted_labels, true_labels), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1(predicted_labels, true_labels), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("test_acc", self.accuracy(predicted_labels, true_labels), prog_bar=True)
        self.log("test_f1", self.f1(predicted_labels, true_labels), prog_bar=True)
        self.log("test_auroc", self.auroc(predicted_labels, true_labels), prog_bar=True)
        self.log("test_prAuc", self.prAuc(predicted_labels, true_labels), prog_bar=True)
        return {'test_acc': self.accuracy.compute(), 'test_f1': self.f1.compute()}

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
