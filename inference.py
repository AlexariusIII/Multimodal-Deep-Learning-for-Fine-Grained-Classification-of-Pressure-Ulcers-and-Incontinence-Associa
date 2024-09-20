import torch
from models.timm_classification import MultiModal_Concat_Timm
from modules.image_classificaiton_module import Image_Classification_Module
from kiadeku_dataset import Image_Classification_Dataset,LightningDataModule,Image_Classification_Dataset_Multilabel
from PIL import Image
from kiadeku_transforms import KiadekuTransforms
import numpy as np
import os
import pandas as pd
from torchmetrics.classification import Accuracy, F1Score, AveragePrecision, AUROC, MulticlassConfusionMatrix, BinaryConfusionMatrix # type: ignore
import lightning as L
import torchvision.transforms as transforms
from kiadeku_transforms import KiadekuTransforms
from collections import Counter
import argparse

class LightningInferenceClassification():
    def __init__(self,ckp_path) :
        super().__init__()
        self.model = Image_Classification_Module.load_from_checkpoint(ckp_path)
        
    def single_image_inference(self, image_path,cat_features):
        image = Image.open(image_path)
        image = self.val_transform(image) # type: ignore
        image = image.to(self.model.device)
        cat_features = cat_features.to(self.model.device)
        image = image.unsqueeze(0)  # Add batch dimension
        cat_features=torch.tensor(cat_features).to(self.model.device)
        # Perform inference
        model.eval() # type: ignore
        with torch.no_grad():
            output = self.model(image,cat_features)
        # Convert output to probabilities (if needed) and get predicted class
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.squeeze().tolist()

    def dm_test_inference(self,dm):
        all_predictions = []
        true_labels = []
        for batch in dm.test_dataloader():
            images, cat_features, true_label  = batch['image'], batch['cat_features'], batch['label']
            images = images.to(self.model.device)
            cat_features = cat_features.to(self.model.device)
            model.eval() # type: ignore
            with torch.no_grad():
                outputs = self.model(images, cat_features)
            probabilities=outputs
            true_labels.extend(true_label)
            all_predictions.extend(probabilities.tolist())
        return all_predictions,true_labels

    def manual_lightning_test(self,test_df):
        image_source = '../data/kiadeku_dataset/annotated_images/'
        test_dataset = Image_Classification_Dataset(image_source=image_source,df=test_df,transform=self.val_transform) # type: ignore
        dm = LightningDataModule(test_dataset=test_dataset)
        trainer = L.Trainer(
                accelerator='gpu',
                devices=[0],
                )
        trainer.test(model=self.model,datamodule=dm)

    def get_metrics_from_preds(self,preds,true_labels,num_classes=2):
        preds = torch.tensor(preds)
        true_labels = torch.tensor([label.item() for label in true_labels])
        accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        return f1(preds, true_labels)

class Kiadeku_Inference():
    def __init__(self) :
        super().__init__()

    def single_image_inference(self, model,image_path,val_transform,cat_features=np.random.rand(5)):
        image = Image.open(image_path)
        image = val_transform(image) # type: ignore
        image = image.to(model.device)
        #cat_features = cat_features.to(model.device) # type: ignore
        image = image.unsqueeze(0)  # Add batch dimension
        cat_features=torch.tensor(cat_features).to(model.device)
        # Perform inference
        model.eval() # type: ignore
        with torch.no_grad():
            outputs = model(image, cat_features)
        # Convert output to probabilities (if needed) and get predicted class
        probabilities = torch.softmax(outputs, dim=1) # type: ignore
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.squeeze().tolist()

    def dm_test_inference(self,dm,model,mode='multiclass'):
        all_predictions = []
        true_labels = []
        for batch in dm.test_dataloader():
            images, cat_features, true_label  = batch['image'], batch['cat_features'], batch['label']
            images = images.to(model.device)
            cat_features = cat_features.to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model(images, cat_features)
            #probabilities=outputs
            if mode == 'multilabel':
                probabilities = torch.sigmoid(outputs)
            else:
                probabilities = torch.softmax(outputs, dim=1)
            true_labels.extend(true_label)
            all_predictions.extend(probabilities.tolist())
        return all_predictions,true_labels

    def dm_test_inference_true_labels(self,dm,model):
        all_predictions = []
        true_labels = []
        for batch in dm.test_dataloader():
            images, cat_features, true_label  = batch['image'], batch['cat_features'], batch['label']
            images = images.to(model.device)
            cat_features = cat_features.to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model(images, cat_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            true_labels.extend(true_label)
            all_predictions.extend(predicted_labels)
        return all_predictions,true_labels

    def get_fold_predictions(self,fold_folder_path,checkpoint_name,dm,fold_num=5,mode='multiclass'):
        preds = []
        for i in range(fold_num):
            CKPT_PATH = f"{fold_folder_path}/fold_{i}/{checkpoint_name}"
            model = Image_Classification_Module.load_from_checkpoint(CKPT_PATH)
            predicted_labels,true_labels = self.dm_test_inference(dm,model,mode=mode)
            #predicted_labels,true_labels = self.dm_test_inference_true_labels(dm,model)
            preds.append(predicted_labels)
        return preds,true_labels

    def average_predictions(self,preds,true_labels,mode='multiclass'):
        ensemble_preds = [np.array(x) for x in preds]
        average_logits = np.mean(ensemble_preds, axis=0)
        preds = torch.tensor(average_logits)
        if mode == 'multilabel':
            true_labels  = torch.stack(true_labels)
        else:
            true_labels = torch.tensor([label.item() for label in true_labels])
        return preds,true_labels

    def mayority_voting(self,preds,true_labels,mode='multiclass'):
        #ensemble_preds = [np.array(x) for x in preds]
        # print(ensemble_preds)
        # average_logits = np.mean(ensemble_preds, axis=0)
        # preds = torch.tensor(average_logits)
        # if mode == 'multilabel':
        #     true_labels  = torch.stack(true_labels)
        # else:
        #     true_labels = torch.tensor([label.item() for label in true_labels])
        return preds,true_labels

    def get_metrices(self,preds,true_labels,num_classes,mode='multiclass'):
        if mode == 'multilabel':
            accuracy = Accuracy(task=mode, num_labels=num_classes)
            f1 = F1Score(task=mode, num_labels=num_classes, average="macro")
            auroc = AUROC(task=mode, num_labels=num_classes)
            prAuc = AveragePrecision(task=mode, num_labels=num_classes,average="macro")
        else:
            mode='multiclass'
            print(f'getting {mode} metrics')
            accuracy = Accuracy(task=mode, num_classes=num_classes)
            f1 = F1Score(task=mode, num_classes=num_classes, average="macro")
            auroc = AUROC(task=mode, num_classes=num_classes)
            prAuc = AveragePrecision(task=mode, num_classes=num_classes,average="macro")

        #ensembled_acc = accuracy(np.argmax(preds, axis=1), true_labels)
        #ensembled_f1 = f1(np.argmax(preds, axis=1), true_labels)
        ensembled_acc = accuracy(preds, true_labels)
        ensembled_f1 = f1(preds, true_labels)
        ensembled_auc = auroc(preds, true_labels)
        ensembled_prAuc = prAuc(preds, true_labels)
        return {'accuracy':ensembled_acc,'f1':ensembled_f1,'auc':ensembled_auc,'prAuc':ensembled_prAuc}

    def get_crossValidationResults(self, preds,true_labels,num_classes,mode='multiclass'):
        scores = {'accuracy':[],'f1':[],'auc':[],'prAuc':[]}
        for pred in preds:
            pred = torch.tensor(np.array(pred),dtype=torch.float64)
            true_labels = torch.tensor(np.array(true_labels))
            torch.set_printoptions(precision=10)
            metrics = self.get_metrices(pred, true_labels,num_classes,mode=mode)
            for key in metrics:
                scores[key].append(metrics[key])
        mean_scores = {}
        for key in scores:
            mean_scores[key]=(np.mean(scores[key]),np.std(scores[key]))
        return mean_scores

    def get_ttaPredictions(self,test_dataloader,transform,model,tta_num=3):
        all_predictions = []
        true_labels = []
        for batch in test_dataloader:
            images, cat_features, true_label  = batch['image'], batch['cat_features'], batch['label']
            probs = []
            images = images.to(model.device)
            cat_features = cat_features.to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model(images, cat_features)
            probabilities = torch.softmax(outputs, dim=1)
            probs.append(probabilities)
            for i in range(tta_num):
                transformed_images = torch.stack([transform(x) for x in images]).to(model.device)
                with torch.no_grad():
                    outputs = model(transformed_images, cat_features)
                probabilities = torch.softmax(outputs, dim=1)
                probs.append(probabilities)
            # for x in probs:
            #     print(x)
            average_prob = torch.mean(torch.stack(probs), dim=0)
            true_labels.extend(true_label)
            all_predictions.extend(average_prob.tolist())
            #break
        return all_predictions,true_labels

def main():
    parser = argparse.ArgumentParser(description='Run image inference on a model.')
    parser.add_argument('--image_path', type=str, nargs='?', default='./example_images/example_image.jpeg',
                        help='Path to the input image (default: ./example_images/example_image.jpg)')
    args = parser.parse_args()

    binary_mapping = {'0': 'PU', '1': 'IAD'}
    iad_mapping = {'1A': 0, '1B': 1, '2A': 2, '2B': 3}
    pu_mapping = {'PU1': 0, 'PU2': 1, 'PU3': 2, 'PU4': 3}
      
    val_transform = KiadekuTransforms(384).val_transforms()

    ckpt_path = './checkpoints/binary/fold_0/tiny_vit_21m_384.ckpt'
    model = Image_Classification_Module.load_from_checkpoint(ckpt_path)
    kia_inference = Kiadeku_Inference()

    pred,probs = kia_inference.single_image_inference(model,args.image_path,val_transform)
    print(f"Prediction: {binary_mapping[str(pred)]}")
    print(f"Confidence: {probs[0]}")


if __name__ == "__main__":
    main()