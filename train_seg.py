import os
import pandas as pd
import numpy as np
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import segmentation_models_pytorch as smp
from kiadeku_dataset import Image_Segmentation_Dataset, LightningDataModule
from kiadeku_transforms import KiadekuTransforms
from modules.binary_semantic_segmentation import BinarySegmentationModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gpu", default='auto', help="GPU selection (comma-separated list of GPU indices or 'auto')")
    args = parser.parse_args()

    # General setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    WORKER_COUNT = 4
    LR = 0.0001
    EPOCHS = 50

    # Dataset setup
    data_source = '../data/kiadeku_dataset/data_pkl/binary'
    image_source = '../data/kiadeku_dataset/annotated_images/'
    mask_source = '../data/kiadeku_dataset/masks/'
    df = pd.read_pickle(os.path.join(data_source, "train_binary.pkl"))

    # Load test Dataset
    train_transform = KiadekuTransforms(img_size=512).train_seg_transform(normalize=True)
    val_transform = KiadekuTransforms(img_size=512).val_seg_transform(normalize=True)
    test_df = pd.read_pickle(os.path.join(data_source, "test_binary.pkl"))
    test_dataset = Image_Segmentation_Dataset(image_source=image_source, mask_source=mask_source, df=test_df, transform=val_transform)
    
    pl.seed_everything(42, workers=True)
    test_iou_scores = []

    for fold in set(df['kfold'].to_list()):
        train_df = df[df['kfold'] != fold]
        val_df = df[df['kfold'] == fold]
        train_dataset = Image_Segmentation_Dataset(image_source, mask_source, train_df, train_transform)
        val_dataset = Image_Segmentation_Dataset(image_source, mask_source, val_df, val_transform)
        dm = LightningDataModule(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_COUNT)
        print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Model setup
        model = smp.Unet(encoder_name='efficientnet-b3', in_channels=3, classes=1)
        module = BinarySegmentationModule(model=model, lr=LR)

        # Logger and callbacks
        wandb_logger = WandbLogger(project="Kiadeku-WoundSegmentation", entity="alexander-brehmer", name=f"effnetb3-Fold-{fold}")
        callbacks = [
            EarlyStopping(monitor="val_iou", patience=5, mode="max", verbose=True),
            ModelCheckpoint(dirpath=os.path.join('checkpoints', f"wound_segmentation/fold_{fold}"), monitor='val_iou', mode='max', save_top_k=1, verbose=True)
        ]

        # Trainer setup
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator=DEVICE,
            devices=-1 if args.gpu == 'auto' else [int(g) for g in args.gpu.split(',')],
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=True
        )

        # Train and test
        trainer.fit(module, datamodule=dm)
        test_result = trainer.test(datamodule=dm, verbose=True)
        test_iou_scores.append(test_result[0]['test_iou'])

        wandb.finish()

    # Calculate and display the average and standard deviation of IoU scores
    average_iou = np.mean(test_iou_scores)
    std_iou = np.std(test_iou_scores)
    print(f"Mean IoU: {average_iou:.4f}, Std Dev IoU: {std_iou:.4f}")

    # Save results
    results_dir = os.path.join('results', 'segmentation')
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame({'average_iou_score': [average_iou], 'std_iou': [std_iou]}).to_csv(os.path.join(results_dir, 'result.csv'), index=False)
    pd.DataFrame({'fold_iou_scores': [test_iou_scores]}).to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)

if __name__ == "__main__":
    main()
