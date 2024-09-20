from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

class KiadekuTransforms():
    def __init__(self,img_size=512):
        self.img_size = img_size

    def rand_aug(self,normalize=True,n=2,m=9):
        t =  transforms.Compose([
            #transforms.RandomResizedCrop(self.img_size),
            transforms.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            transforms.RandAugment(num_ops=n,magnitude=m),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([
                t,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t

    def triv_aug(self,normalize=True):
        t =  transforms.Compose([
            #transforms.RandomResizedCrop(self.img_size),
            transforms.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([
                t,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t

    def soft_transforms(self,normalize=True):
        t =  transforms.Compose([
            #transforms.RandomResizedCrop(self.img_size),
            transforms.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomRotation(degrees=5),    
            transforms.ToTensor(),    
        ])
        if normalize:
            t = transforms.Compose([
                t,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t

    def heavy_transforms(self, normalize=True):
        t = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Increase rotation angle
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Increase intensity
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Affine transformation
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Perspective transformation
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([
                t,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t

    
    def val_transforms(self,normalize=True):
        t =  transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            #transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([
                t,
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t
    
    def train_seg_transform(self,normalize=True):
        t =  v2.Compose([
            #v2.RandomResizedCrop(512),
            v2.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(degrees=40), # type: ignore
            v2.ToTensor(),
        ])
        if normalize:
            t = v2.Compose([
                t,
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t
    
    def val_seg_transform(self,normalize=True):
        t =  v2.Compose([
            v2.Resize((self.img_size, self.img_size),interpolation=InterpolationMode.BICUBIC),
            v2.ToTensor(),
        ])
        if normalize:
            t = v2.Compose([
                t,
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return t

