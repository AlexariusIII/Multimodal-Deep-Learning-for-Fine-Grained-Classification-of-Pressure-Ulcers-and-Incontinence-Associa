import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
#from torchmetrics.detection import IntersectionOverUnion
import segmentation_models_pytorch as smp
import wandb
import numpy as np

class BinarySegmentationModule(L.LightningModule):
    def __init__(self, model,lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
        self.lr = lr
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.sigmoid(logits) > 0.5
        iou = self.compute_iou(preds, masks)
        self.log('val_loss', loss)
        self.log('val_iou', iou)

        #Visualize Masks in wandb
        #print(np.unique(masks[0].cpu().flatten()))
        #print(np.unique(preds[0].cpu().flatten()))
        # log_gt_mask = masks[0].permute(1, 2, 0).cpu().numpy().squeeze()
        # log_pred_mask = preds[0].permute(1, 2, 0).cpu().numpy().squeeze()
        # print(log_pred_mask)
        # log_image =  wandb.Image(images[0],masks={
        #     "pred": {"mask_data": log_pred_mask,
        #                     "class_labels": {0:'bg',1:'mask'},
        #                     },
        #     "ground_truth": {"mask_data": log_gt_mask,
        #                     "class_labels": {0:'bg',1:'mask'},
        #                     },
        # },)
        # wandb.log({"Image": log_image})

 
    def test_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.sigmoid(logits) > 0.5
        iou = self.compute_iou(preds, masks)
        self.log('test_loss', loss)
        self.log('test_iou', iou)

        return {'test_iou':iou}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def compute_iou(self, preds, target):
        intersection = torch.logical_and(target, preds).sum(dim=(1, 2)).float()
        union = torch.logical_or(target, preds).sum(dim=(1, 2)).float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()