from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from pytorch_lightning import LightningDataModule

class ImageClassificationDataset(Dataset):
    """
    A dataset class for image classification tasks, supporting both multiclass and multilabel classification.
    """
    def __init__(self, image_source: str, df: pd.DataFrame, transform=None):
        """
        Args:
            image_source (str): Directory where image files are stored.
            df (pd.DataFrame): DataFrame containing image file names and labels.
            transform: Transformations to be applied on the images.
        """
        self.image_source = image_source
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_source, str(self.df.iloc[idx, 0]))
        image = Image.open(image_path).convert('RGB')
        if 'multilabel' in self.df.columns:
            label = self.df.iloc[idx, self.df.columns.str.startswith('WP_')].tolist()
        else:
            label = self.df.iloc[idx, 1]
        cat_features = np.array(self.df.iloc[idx, 2:7], dtype=np.float32)  # Adjust index based on actual layout

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': torch.tensor(label), 'cat_features': cat_features}

    def get_id(self, idx):
        return self.df.iloc[idx, 0]

    def get_class_distribution(self):
        return self.df['label'].value_counts().to_dict()

    def get_class_num(self):
        return self.df['label'].nunique()

class ImageSegmentationDataset(Dataset):
    """
    A dataset class for image segmentation tasks.
    """
    def __init__(self, image_source: str, mask_source: str, df: pd.DataFrame, transform=None):
        self.image_source = image_source
        self.mask_source = mask_source
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_source, str(self.df.iloc[idx, 0]))
        mask_path = os.path.join(self.mask_source, str(self.df.iloc[idx, 0]))
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L').point(lambda x: 1 if x > 0 else 0)
        label = self.df.iloc[idx, 1]

        if self.transform:
            image, mask = self.transform(image, mask)

        return {'image': image, 'mask': mask, 'label': torch.tensor(label)}

    def get_class_num(self):
        return 1  # For segmentation, typically binary mask

class LightningDatasetModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling training, validation, and test data loaders for datasets.
    """
    def __init__(self, train_dataset=None, valid_dataset=None, test_dataset=None, batch_size=64, num_workers=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Optionally print dataset summaries here
        print(f"Train Dataset: {len(self.train_dataset)}")
        print(f"Valid Dataset: {len(self.valid_dataset)}")
        print(f"Test Dataset: {len(self.test_dataset)}")
        print(f"Number of classes: {self.get_class_num()}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_class_num(self):
        if self.train_dataset:
            return self.train_dataset.get_class_num()
        elif self.valid_dataset:
            return self.valid_dataset.get_class_num()
        elif self.test_dataset:
            return self.test_dataset.get_class_num()
        return None

def main():
    print('DataModule setup complete.')

if __name__ == "__main__":
    main()
