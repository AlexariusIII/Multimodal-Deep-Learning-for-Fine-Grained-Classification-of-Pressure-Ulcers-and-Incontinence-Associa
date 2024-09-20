from torch import nn
import torch
import timm

#Setup Model
class MultiModal_Concat_Timm(torch.nn.Module):
    def __init__(self,arch='resnet50', num_cat_feature=5, num_output_classes=2, multi_modal=False, device='gpu') :
        super().__init__()
        self.device = device
        self.multi_modal = multi_modal
        self.cat_feature_num = num_cat_feature
        self.num_output_classes = num_output_classes
        self.image_encoder = timm.create_model(arch, pretrained=True, num_classes=0)

        #self.vocab_size = 9*5
        #self.vocab_size =5
        self.vocab_size = 30
        self.num_inputs = num_cat_feature
        self.embedding_dim = 256
        # self.cat_encoder = nn.Sequential(nn.Embedding(self.vocab_size, self.embedding_dim),nn.ReLU() ,nn.Linear(self.embedding_dim, 2*self.embedding_dim),nn.ReLU(),nn.Linear(2*self.embedding_dim, self.embedding_dim),nn.ReLU() )
        # self.cat_encoder = nn.Sequential(
        #     nn.Embedding(self.vocab_size, self.embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embedding_dim, 2*self.embedding_dim),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.ReLU(),
        #     nn.Linear(2*self.embedding_dim, self.embedding_dim),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.ReLU() )
        self.cat_encoder = nn.Sequential(nn.Embedding(self.vocab_size, 2*self.embedding_dim),nn.ReLU() ,nn.Linear(2*self.embedding_dim, self.embedding_dim),nn.ReLU() )
        # self.cat_encoder = nn.Sequential(
        #     nn.Embedding(self.vocab_size, self.embedding_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embedding_dim, 2 * self.embedding_dim),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embedding_dim * 2, self.embedding_dim),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2 * self.embedding_dim, self.embedding_dim),  # Additional layer
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        # )
        self.intermediate = torch.tensor([[0,5,10,15,20]])
        # Define output layers
        if self.multi_modal:
            self.output_layer = nn.Linear(self.image_encoder.num_features + self.num_inputs * self.embedding_dim, self.num_output_classes)
        else:
            self.output_layer = nn.Linear(self.image_encoder.num_features, self.num_output_classes)
    
    def forward(self, img, cat):
        # Output shape is [Batchsize,  num_features]
        img_features = self.image_encoder(img)
        if self.multi_modal:
            self.intermediate = self.intermediate.to(cat)
            cat = (cat + self.intermediate).long()
           
            #cat = cat.long()
            # Output shape is [Batchsize, num_inputs, num_features]
            cat_features = self.cat_encoder(cat)
            cat_features_flattened = torch.flatten(cat_features, start_dim=1)
            concat_features = torch.cat([img_features, cat_features_flattened], dim=1)
        else:
            concat_features = img_features
        
        out = self.output_layer(concat_features)
        return out
    
    def get_name(self):
        return f"{self.image_encoder.default_cfg['architecture']}"
