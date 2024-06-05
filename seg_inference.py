import torch
import numpy as np
from kiadeku_dataset import LightningDataModule
from modules.binary_semantic_segmentation import BinarySegmentationModule
import lightning as L

class Kiadeku_SegInference:
    """
    Class to handle inference tasks for image segmentation models, 
    providing utilities for computing IoU, processing single images, and handling test predictions across folds.
    """
    def __init__(self):
        super().__init__()

    def compute_iou(self, preds, target):
        """
        Compute the Intersection over Union (IoU) between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted binary masks.
            target (torch.Tensor): Ground truth binary masks.

        Returns:
            float: Mean IoU across all predictions and targets.
        """
        intersection = torch.logical_and(target, preds).sum(dim=(1, 2)).float()
        union = torch.logical_or(target, preds).sum(dim=(1, 2)).float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def fold_iou_and_mean_iou_from_fold_preds(self, all_fold_preds, true_masks):
        """
        Compute the IoU for each fold and the mean IoU across all folds.

        Args:
            all_fold_preds (list): List of predictions from each fold.
            true_masks (list): List of true masks corresponding to the predictions.

        Returns:
            tuple: Tuple containing list of IoUs for each fold and the mean IoU.
        """
        torch.set_printoptions(precision=10)
        iou_scores = [self.compute_iou(torch.tensor(preds), torch.tensor(true_masks)) for preds in all_fold_preds]
        mean_iou = torch.mean(torch.stack(iou_scores))
        return iou_scores, mean_iou

    def get_fold_test_predictions_and_true_masks(self, dm, fold_folder_path, checkpoint_name):
        """
        Fetch predictions and true masks for all folds given a data module and checkpoint details.

        Args:
            dm (LightningDataModule): The data module with the test dataset.
            fold_folder_path (str): Path to the directory containing fold checkpoints.
            checkpoint_name (str): Name of the checkpoint file.

        Returns:
            tuple: Tuple containing list of all predictions and true masks for the folds.
        """
        all_fold_preds = []
        true_masks = []
        for i in range(5):  # Assuming 5 folds
            CKPT_PATH = f"{fold_folder_path}/fold_{i}/{checkpoint_name}"
            model = BinarySegmentationModule.load_from_checkpoint(CKPT_PATH)
            model.eval()
            fold_preds = []
            for batch in dm.test_dataloader():
                images, masks = batch['image'], batch['mask']
                images = images.to(model.device)
                with torch.no_grad():
                    outputs = model(images)
                pred = torch.sigmoid(outputs) > 0.5
                fold_preds.extend(pred.tolist())
                true_masks.extend(masks.tolist())

            all_fold_preds.append(fold_preds)
        return all_fold_preds, true_masks

    def get_average_test_predictions_from_folds(self, dm, fold_folder_path, checkpoint_name):
        """
        Calculate average predictions from model checkpoints across folds.

        Args:
            dm (LightningDataModule): Data module with test data.
            fold_folder_path (str): Path to the folder containing checkpoints.
            checkpoint_name (str): Name of the checkpoint file.

        Returns:
            tuple: Tuple containing average predictions and true masks.
        """
        all_fold_outputs = []
        true_masks = []
        for i in range(2):  # Assuming averaging across 2 folds for simplicity
            CKPT_PATH = f"{fold_folder_path}/fold_{i}/{checkpoint_name}"
            model = BinarySegmentationModule.load_from_checkpoint(CKPT_PATH)
            model.eval()
            fold_outputs = []
            for batch in dm.test_dataloader():
                images, masks = batch['image'], batch['mask']
                images = images.to(model.device)
                with torch.no_grad():
                    outputs = model(images)
                fold_outputs.extend(outputs.tolist())
                true_masks.extend(masks.tolist())
            all_fold_outputs.append(fold_outputs)

        average_logits = np.mean([np.array(x) for x in all_fold_outputs], axis=0)
        average_preds = torch.sigmoid(torch.tensor(average_logits)) > 0.5
        return average_preds, torch.tensor(true_masks)

    def single_image_segmentation(self, ckpt_path, image):
        """
        Perform segmentation on a single image using a trained model.

        Args:
            ckpt_path (str): Path to the model checkpoint.
            image (PIL.Image or torch.Tensor): Image to segment.

        Returns:
            torch.Tensor: Predicted mask for the image.
        """
        model = BinarySegmentationModule.load_from_checkpoint(ckpt_path)
        model.eval()
        image = image.to(model.device)
        with torch.no_grad():
            outputs = model(image)
        pred_mask = torch.sigmoid(outputs) > 0.5
        return pred_mask

def main():
    print('Segmentation Inference Setup Complete')

if __name__ == "__main__":
    main()
