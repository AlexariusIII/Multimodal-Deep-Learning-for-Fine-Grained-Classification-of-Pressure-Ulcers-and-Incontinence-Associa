from torchmetrics.classification import Accuracy, F1Score, AveragePrecision,AUROC
import torch
from torch import nn
import lightning as L
import torch.optim.lr_scheduler as lr_scheduler

class Image_Classification_Module(L.LightningModule):
    def __init__(self, model,lr,num_classes):
        super().__init__()
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=['model'])
        self.model = model
        #self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.prAuc = AveragePrecision(task="multiclass", num_classes=num_classes, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, image, cat_features):
        return self.model(image, cat_features)

    def _shared_step(self, batch, train=False):
        images = batch['image']
        true_labels = batch['label']
        cat_features = batch['cat_features']
        logits = self(images,cat_features)
        loss = self.loss(logits,true_labels)
        #predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels=logits
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch,train=True)
        train_acc = self.accuracy(predicted_labels, true_labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        val_acc = self.accuracy(predicted_labels, true_labels)
        val_f1 = self.f1(predicted_labels, true_labels)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", val_f1, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        #print(predicted_labels,true_labels)
        test_acc = self.accuracy(predicted_labels, true_labels)
        test_f1 = self.f1(predicted_labels, true_labels)
        test_auroc = self.auroc(predicted_labels, true_labels)
        test_prAuc = self.prAuc(predicted_labels, true_labels)

        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_f1", test_f1, prog_bar=True)
        self.log("test_auroc", test_auroc, prog_bar=True)
        self.log("test_prAuc", test_prAuc, prog_bar=True)

        return {'test_acc':test_acc,'test_f1':test_f1,'test_auroc':test_auroc,'test_prAuc':test_prAuc}

    def configure_optimizers(self):
         return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

class Image_Classification_Module_MultiLabel(L.LightningModule):
    def __init__(self, model,lr,num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="multilabel", num_labels=num_classes)
        self.prAuc = AveragePrecision(task="multilabel", num_labels=num_classes, average="macro")
        self.f1 = F1Score(task="multilabel", num_labels=num_classes, average="macro")
        self.auroc = AUROC(task="multilabel", num_labels=num_classes)

    def forward(self, image, cat_features):
        return self.model(image, cat_features)

    def _shared_step(self, batch, train=False):
        images = batch['image']
        true_labels = batch['label']
        cat_features = batch['cat_features']
        logits = self(images,cat_features)
        loss = self.loss(logits,true_labels.float())
        #predicted_labels = torch.argmax(logits, dim=1)
        sigmoid_logits = torch.sigmoid(logits)
        predicted_labels = (sigmoid_logits > 0.5).float() 
        #predicted_labels=logits
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch,train=True)
        train_acc = self.accuracy(predicted_labels, true_labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        val_acc = self.accuracy(predicted_labels, true_labels)
        val_f1 = self.f1(predicted_labels, true_labels)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", val_f1, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        test_acc = self.accuracy(predicted_labels, true_labels)
        test_f1 = self.f1(predicted_labels, true_labels)
        test_auroc = self.auroc(predicted_labels, true_labels)
        test_prAuc = self.prAuc(predicted_labels, true_labels)

        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_f1", test_f1, prog_bar=True)
        self.log("test_auroc", test_auroc, prog_bar=True)
        self.log("test_prAuc", test_prAuc, prog_bar=True)

        return {'test_acc':test_acc,'test_f1':test_f1,'test_auroc':test_auroc,'test_prAuc':test_prAuc}

    def configure_optimizers(self):
         return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
