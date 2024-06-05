from torchvision import transforms
from torchvision.transforms import InterpolationMode

class KiadekuTransforms:
    """
    Provides customizable transformation pipelines for image preprocessing, tailored for different
    training and validation scenarios including normalization options.
    """
    def __init__(self, img_size=512):
        """
        Initialize the transformation settings with a default or specified image size.

        Args:
            img_size (int): Target size for resizing images.
        """
        self.img_size = img_size

    def rand_aug(self, normalize=True, n=2, m=9):
        """
        Random augmentation for data augmentation during training.

        Args:
            normalize (bool): Whether to normalize images.
            n (int): Number of random augmentation operations to apply.
            m (int): Magnitude for random augmentations.

        Returns:
            torchvision.transforms.Compose: Composed transformations including normalization if enabled.
        """
        t = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandAugment(num_ops=n, magnitude=m),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([t, self._get_normalization()])
        return t

    def triv_aug(self, normalize=True):
        """
        Apply trivial augmentation for more subtle transformations during training.

        Args:
            normalize (bool): Whether to normalize images.

        Returns:
            torchvision.transforms.Compose: Composed transformations including normalization if enabled.
        """
        t = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([t, self._get_normalization()])
        return t

    def soft_transforms(self, normalize=True):
        """
        Apply softer transformations suitable for less aggressive augmentation needs.

        Args:
            normalize (bool): Whether to normalize images.

        Returns:
            torchvision.transforms.Compose: Composed transformations including normalization if enabled.
        """
        t = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([t, self._get_normalization()])
        return t

    def heavy_transforms(self, normalize=True):
        """
        Apply heavy transformations for extensive data augmentation.

        Args:
            normalize (bool): Whether to normalize images.

        Returns:
            torchvision.transforms.Compose: Composed transformations including normalization if enabled.
        """
        t = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([t, self._get_normalization()])
        return t

    def val_transforms(self, normalize=True):
        """
        Transformations for validation data.

        Args:
            normalize (bool): Whether to normalize images.

        Returns:
            torchvision.transforms.Compose: Composed transformations including normalization if enabled.
        """
        t = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        if normalize:
            t = transforms.Compose([t, self._get_normalization()])
        return t

    def _get_normalization(self):
        """
        Private method to get the normalization transformation based on standard ImageNet stats.

        Returns:
            torchvision.transforms.Normalize: Normalization transformation.
        """
        return transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Example of using the class
def main():
    kt = KiadekuTransforms(img_size=256)
    transform = kt.rand_aug()
    print("Random Augmentation Pipeline:", transform)

if __name__ == "__main__":
    main()
