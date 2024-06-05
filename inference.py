import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from models.timm_classification import MultiModal_Concat_Timm
from modules.image_classification_module import Image_Classification_Module
from datasets.kiadeku_dataset import Image_Classification_Dataset, LightningDataModule, Image_Classification_Dataset_Multilabel
from metrics.torchmetrics import Accuracy, F1Score, AveragePrecision, AUROC

class LightningInferenceClassification(LightningModule):
    """
    A class for handling image classification inference using pre-trained models from PyTorch Lightning checkpoints.
    """
    def __init__(self, ckp_path):
        super().__init__()
        self.model = Image_Classification_Module.load_from_checkpoint(ckp_path)
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def single_image_inference(self, image_path, cat_features):
        image = Image.open(image_path)
        image = self.val_transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        cat_features = torch.tensor(cat_features).float().unsqueeze(0)  # Process and add batch dimension
        image, cat_features = image.to(self.device), cat_features.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image, cat_features)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.squeeze().tolist()

    def dm_test_inference(self, dm):
        all_predictions = []
        true_labels = []
        self.model.eval()
        for batch in dm.test_dataloader():
            images, cat_features, labels = batch['image'], batch['cat_features'], batch['label']
            images, cat_features = images.to(self.device), cat_features.to(self.device)
            with torch.no_grad():
                outputs = self.model(images, cat_features)
            probabilities = outputs
            true_labels.extend(labels)
            all_predictions.extend(probabilities.tolist())
        return all_predictions, true_labels

    def manual_lightning_test(self, test_df):
        image_source = '../data/kiadeku_dataset/annotated_images/'
        test_dataset = Image_Classification_Dataset(image_source=image_source, df=test_df, transform=self.val_transform)
        dm = LightningDataModule(test_dataset=test_dataset)
        trainer = Trainer(accelerator='gpu', devices=[0])
        trainer.test(model=self.model, datamodule=dm)

    def get_metrics_from_preds(self, preds, true_labels, num_classes=2):
        preds = torch.tensor(preds)
        true_labels = torch.tensor([label.item() for label in true_labels])
        accuracy = Accuracy(num_classes=num_classes)
        f1 = F1Score(num_classes=num_classes, average="macro")
        return f1(preds, true_labels)