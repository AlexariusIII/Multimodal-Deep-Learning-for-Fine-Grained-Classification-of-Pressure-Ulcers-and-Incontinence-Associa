from torch import nn
import torch
import timm

class MultiModal_Concat_Timm(nn.Module):
    """
    A PyTorch module for a multimodal neural network that integrates image features from a pretrained
    model and categorical features via embedding layers. The network supports both unimodal (image-only)
    and multimodal (image and categorical data) operations.

    Attributes:
        device (str): The device on which the model will be running ('cpu' or 'gpu').
        multi_modal (bool): Flag to determine whether the model operates in multimodal mode.
        cat_feature_num (int): Number of categorical features.
        num_output_classes (int): Number of output classes for the final classifier.
        image_encoder (nn.Module): A CNN encoder for image processing, taken from the TIMM library.
        vocab_size (int): The number of unique categories for the embedding layer.
        num_inputs (int): Number of input categorical features.
        embedding_dim (int): Dimensionality of the embedding for categorical features.
        cat_encoder (nn.Module): A sequential model for encoding categorical features.
        intermediate (torch.Tensor): An intermediate tensor used for categorical feature transformation.
        output_layer (nn.Linear): The final linear layer that outputs the class logits.

    Args:
        arch (str): Model architecture from TIMM library used for image encoding.
        num_cat_feature (int): Number of categorical features expected in the input.
        num_output_classes (int): Number of classes for the output classification.
        multi_modal (bool): Whether the model should use both image and categorical inputs.
        device (str): The computing device ('cpu' or 'gpu').

    Methods:
        forward(img, cat):
            Defines the forward pass of the model.
            
            Args:
                img (torch.Tensor): Input tensor containing batch of images.
                cat (torch.Tensor): Input tensor containing batch of categorical features.
            
            Returns:
                torch.Tensor: The output logits from the model.

        get_name():
            Returns the architecture name of the image encoder.
    """

    def __init__(self, arch='resnet50', num_cat_feature=5, num_output_classes=2, multi_modal=False, device='gpu'):
        super().__init__()
        self.device = device
        self.multi_modal = multi_modal
        self.cat_feature_num = num_cat_feature
        self.num_output_classes = num_output_classes
        self.image_encoder = timm.create_model(arch, pretrained=True, num_classes=0)

        self.vocab_size = 30
        self.num_inputs = num_cat_feature
        self.embedding_dim = 256
        self.cat_encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, 2 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU()
        )
        self.intermediate = torch.tensor([[0, 5, 10, 15, 20]])

        if self.multi_modal:
            self.output_layer = nn.Linear(self.image_encoder.num_features + self.num_inputs * self.embedding_dim, self.num_output_classes)
        else:
            self.output_layer = nn.Linear(self.image_encoder.num_features, self.num_output_classes)

    def forward(self, img, cat):
        img_features = self.image_encoder(img)
        if self.multi_modal:
            self.intermediate = self.intermediate.to(cat)
            cat = (cat + self.intermediate).long()
            cat_features = self.cat_encoder(cat)
            cat_features_flattened = torch.flatten(cat_features, start_dim=1)
            concat_features = torch.cat([img_features, cat_features_flattened], dim=1)
        else:
            concat_features = img_features

        out = self.output_layer(concat_features)
        return out

    def get_name(self):
        return f"{self.image_encoder.default_cfg['architecture']}"
