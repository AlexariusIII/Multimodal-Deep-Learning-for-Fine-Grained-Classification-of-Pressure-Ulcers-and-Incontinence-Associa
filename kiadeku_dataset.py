from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L
import torch
import os
from PIL import Image
import pandas as pd
import lightning as L
import numpy as np

class Image_Classification_Dataset(Dataset):
    def __init__(self, image_source: str, df :pd.DataFrame ,transform=None) -> None:
        self.image_source=image_source
        self.df = df
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_source, str(self.df.iloc[idx, 0]))
        image = Image.open(image_path).convert('RGB')
        label  = self.df.iloc[idx, 1]
        cat_features = np.array(self.df.iloc[idx][['SB_Lokalisation','SB_Mobilitaet', 'SB_Wahrnehmungsfaehigkeit', 'SB_Urinkontinenz','SB_Stuhlkontinenz']].values, 
                            dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': torch.tensor(label), 'cat_features':cat_features}
        
    def get_id(self, idx):
        return self.df.iloc[idx, 0]
    
    def get_class_num(self):
        return self.df['label'].nunique()
    
    def get_class_distribution(self):
        return self.df['label'].value_counts().to_dict()

class Image_Classification_Dataset_Multilabel(Dataset):
    def __init__(self, image_source: str, df :pd.DataFrame ,transform=None) -> None:
        self.image_source=image_source
        self.df = df
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_source, str(self.df.iloc[idx, 0]))
        image = Image.open(image_path).convert('RGB')
        labels = self.df.iloc[idx, self.df.columns.str.startswith('WP_')].tolist()
        cat_features = np.array(self.df.iloc[idx][['SB_Lokalisation','SB_Mobilitaet', 'SB_Wahrnehmungsfaehigkeit', 'SB_Urinkontinenz','SB_Stuhlkontinenz']].values, 
                            dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': torch.tensor(labels), 'cat_features':cat_features}
        
    def get_id(self, idx):
        return self.df.iloc[idx, 0]
    
    def get_class_num(self):
        return len(self.df.columns[self.df.columns.str.startswith('WP_')])
        #return len(self.df.columns[6:-1].to_list())
    
    def get_class_distribution(self):
        return self.df['label'].value_counts().to_dict()

class Image_Segmentation_Dataset(Dataset):
    def __init__(self, image_source: str,mask_source:str,df :pd.DataFrame ,transform=None) -> None:
        self.image_source=image_source
        self.mask_source=mask_source
        self.df = df
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_source, str(self.df.iloc[idx, 0]))
        image = Image.open(image_path).convert('RGB')
        mask_path = os.path.join(self.mask_source, str(self.df.iloc[idx, 0]))
        mask = Image.open(mask_path).convert('L')

        #convet 255 pixel to 1
        mask = np.array(mask)
        mask[mask == 255] = 1
        mask[mask != 0] = 1
        mask = Image.fromarray(mask)
        label  = self.df.iloc[idx, 1]

        if self.transform:
            image,mask = self.transform(image,mask)
            mask[mask != 0] = 1
        return {'image': image,'mask':mask,'label': torch.tensor(label)}
        
    def get_id(self, idx):
        return self.df.iloc[idx, 0]
    
    def get_class_distribution(self):
        return self.df['label'].value_counts().to_dict()
    
    def get_class_num(self):
        return 1

class LightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        batch_size=64,
        num_workers=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def prepare_data(self):
        return

    def setup(self, stage=None):
        if self.train_dataset:
            print(f"Train Dataset: {len(self.train_dataset)}")
        if self.valid_dataset:
            print(f"Valid Dataset: {len(self.valid_dataset)}")
        if self.test_dataset:
            print(f"Test Dataset: {len(self.test_dataset)}")
        print(f"Number of classes: {self.get_class_num()}")

    def get_class_num(self):
        if self.train_dataset:
            return self.train_dataset.get_class_num()
        if self.valid_dataset:
            return self.valid_dataset.get_class_num()
        if self.test_dataset:
            return self.test_dataset.get_class_num()
        return None
        
    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self,batch_size=None):
        if batch_size == None:
            batch_size = len(self.test_dataset)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader