from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, yaml, pickle
from kiadeku_dataset import Image_Classification_Dataset,LightningDataModule,Image_Classification_Dataset_Multilabel
from kiadeku_transforms import KiadekuTransforms
import pandas as pd
import torch
from models.timm_classification import MultiModal_Concat_Timm
from modules.image_classificaiton_module import Image_Classification_Module,Image_Classification_Module_MultiLabel
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
import numpy as np
from inference import Kiadeku_Inference


def main():
    torch.set_float32_matmul_precision('high')
    #load config and args
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default='config', help="Select config")
    parser.add_argument("-d", "--dataset", default='binary', help="Select Dataset")
    parser.add_argument("-m", "--multimodal", default='False', help="Set Multimodality")
    parser.add_argument("-cr", "--crop", default='False', help="Select Cropped Images")
    parser.add_argument("-o", "--oversample", default='False', help="Turn Oversampling to 1:1 Ratio on")
    parser.add_argument("-g", "--gpu", default='auto', help="Select GPU")
    parser.add_argument("-t", "--transforms", default='none', help="Select Transforms")
    parser.add_argument("-a", "--architecture", default='convnext', help="Select Architecture")

    args = vars(parser.parse_args())
    config_path = os.path.join("configs", args['config'] + ".yaml")
    multimodality = args['multimodal'] == 'True'
    cropping = args['crop'] == 'True'
    oversample = args['oversample'] =='True'
    gpu = [int(g) for g in args['gpu']] if args['gpu'] != "auto" else "auto"

    with open(config_path, "r") as conf:
            config = yaml.safe_load(conf)

    lr = config['Training']['lr']
    epochs = config['Training']['epochs']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Learning rate: {lr}, Epochs: {epochs}, Device: {device}")

    #Load dataframes and mapping
    data_source = '../data/kiadeku_dataset/data_pkl'
    if cropping == False:
        image_source = '../data/kiadeku_dataset/annotated_images/'
    else:
        print('Using cropped images')
        image_source = '../data/kiadeku_dataset/cropped_images/'    

    class_mapping = pickle.load(open('../data/kiadeku_dataset/class_mappings.pkl', 'rb'))

    #Define if multiclass or multilabel classification
    multi_label_datasets = ['wundform','wundgrundtyp','wundrand_eigenschaft','wundumgebung','wundumgebungsfarbe']
    mode = 'multiclass'
    if args['dataset'] in multi_label_datasets:
            mode = 'multilabel'
    #Read Train and Test DF
    df_folder = os.path.join(data_source, args['dataset'])
    df = pd.read_pickle(os.path.join(df_folder,f"train_{args['dataset']}.pkl"))
    test_df = pd.read_pickle(os.path.join(df_folder,f"test_{args['dataset']}.pkl"))
    if oversample == True:
        print('using oversampled dataset')
        df = pd.read_pickle(os.path.join(os.path.join('../data/kiadeku_dataset/oversampled/data_pkl', args['dataset']),f"train_{args['dataset']}.pkl"))
    
    test_f1_scores = []
    test_auroc_scores = []
    test_prAuc_scores = []
    L.seed_everything(42, workers=True)
    print(f"Begin Training on dataset: {args['dataset']}, On Gpu : {args['gpu']}, Multimodality: {multimodality}, Mode: {mode}, Oversample: {oversample}, Cropping: {cropping}")
    for fold in set(df['kfold'].to_list()):    
        print(f"Start Training Fold: {fold}")
        train_df = df[df['kfold'] != fold]
        val_df = df[df['kfold'] == fold]

        #Setup transforms, datasets and datamodule
        transforms = {
                'none': KiadekuTransforms(config['Augs']['image_size']).val_transforms(),
                'soft' : KiadekuTransforms(config['Augs']['image_size']).soft_transforms(),
                'heavy' : KiadekuTransforms(config['Augs']['image_size']).heavy_transforms(),
                'randaug': KiadekuTransforms(config['Augs']['image_size']).rand_aug(),
                'randaug_strong': KiadekuTransforms(config['Augs']['image_size']).rand_aug(n=4,m=12),
                'triv': KiadekuTransforms(config['Augs']['image_size']).triv_aug(),
        }
        train_transform = transforms[args['transforms']]
        val_transform = KiadekuTransforms(config['Augs']['image_size']).val_transforms()

        if mode == 'multilabel':
                train_dataset = Image_Classification_Dataset_Multilabel(image_source=image_source,df=train_df,transform=train_transform)
                val_dataset = Image_Classification_Dataset_Multilabel(image_source=image_source,df=val_df,transform=val_transform)
                test_dataset = Image_Classification_Dataset_Multilabel(image_source=image_source,df=test_df,transform=val_transform)
        else:
                train_dataset = Image_Classification_Dataset(image_source=image_source,df=train_df,transform=train_transform)
                val_dataset = Image_Classification_Dataset(image_source=image_source,df=val_df,transform=val_transform)
                test_dataset = Image_Classification_Dataset(image_source=image_source,df=test_df,transform=val_transform)

        dm = LightningDataModule(train_dataset, val_dataset,test_dataset, batch_size=config['Training']['batch_size'],num_workers=config['Training']['worker_count'])
        print(f"Train Size: {len(train_dataset)}, Val Size:{len(val_dataset)}, Test Size: {len(test_dataset)}")

        #Setup model and train params
        img_encoders = {
                'tinyvit':'tiny_vit_21m_384.dist_in22k_ft_in1k', #21.23	
                'convnext':'convnextv2_tiny.fcmae_ft_in22k_in1k_384', #28.6
                'caformer':'caformer_s18.sail_in22k_ft_in1k_384', #26.3
                'efficientnetV2' :'tf_efficientnetv2_s.in21k_ft_in1k', #21.5
        }

        num_classes = dm.get_class_num()
        print(f"Setting up model with architecture: {args['architecture']} and {num_classes} Classes")
        model = MultiModal_Concat_Timm(arch=img_encoders[args['architecture']], num_cat_feature=5,num_output_classes=num_classes,multi_modal=multimodality,device=device) # type: ignore
        #Load lightning module
        module = Image_Classification_Module(model,lr, num_classes)
        if mode == 'multilabel':
                module = Image_Classification_Module_MultiLabel(model,lr, num_classes)
        
        #Initiate logger
        #test_dim = '512'
        #group_name = f"{args['architecture']}-T: {args['transforms']}-M:{multimodality}-Cr:{cropping}-O:{oversample}-test_dim{test_dim}"
        group_name = f"{args['architecture']}-T: {args['transforms']}-M:{multimodality}-Cr:{cropping}-O:{oversample}"
        file_name =  f"{model.get_name()}-Fold: {fold}"
        wandb_logger = WandbLogger(entity='alexander-brehmer', project=f"Kiadeku-{args['dataset']}", group=group_name, name=file_name)
        
        #Train
        #ckpt_path=os.path.join('checkpoints',f"{args['dataset']}/{args['architecture']}/t_{args['transforms']}_m_{multimodality}_cr_{cropping}_o_{oversample}-test_dim{test_dim}/fold_{fold}")
        ckpt_path=os.path.join('checkpoints',f"{args['dataset']}/{args['architecture']}/t_{args['transforms']}_m_{multimodality}_cr_{cropping}_o_{oversample}/fold_{fold}")
        early_stop_callback = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=15, verbose=True, mode="max")
        checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode = 'max',
        dirpath=ckpt_path,
        filename=f"{model.get_name()}",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=True,
        verbose=True ,
        )
        #print(device)
        #print(find_usable_cuda_devices(2))
        #print(torch.cuda.get_device_name())
        # print(torch.cuda.current_device())
        # print(torch.cuda.device_count())
        trainer = L.Trainer(
                check_val_every_n_epoch=1,
                #enable_checkpointing=False,
                max_epochs=epochs,
                accelerator=device,
                #devices=gpu,
                devices=-1,
                logger=wandb_logger,
                deterministic=True,
                callbacks=[early_stop_callback,checkpoint_callback],
                )
        trainer.fit(model=module, datamodule=dm)
        test_res = trainer.test(datamodule=dm,ckpt_path=os.path.join(ckpt_path,f"{model.get_name()}.ckpt"),verbose=True)
        #test_res = trainer.test(datamodule=dm,ckpt_path='best',verbose=True)
        wandb.finish()
        print(test_res)
        test_f1_scores.append(test_res[0]['test_f1'])
        test_auroc_scores.append(test_res[0]['test_auroc'])
        test_prAuc_scores.append(test_res[0]['test_prAuc'])
 
    print(f"F1:Scores: {test_f1_scores}")
    print(f"AUROC:Scores: {test_auroc_scores}")
    print(f"PR_Auc:Scores: {test_prAuc_scores}")
    average_f1 = np.mean(test_f1_scores)
    average_auroc = np.mean(test_auroc_scores)
    average_prAuc = np.mean(test_prAuc_scores)
    std_deviation_f1 = np.std(test_f1_scores)
    std_deviation_auroc = np.std(test_auroc_scores)
    std_deviation_prAuc = np.std(test_prAuc_scores)
    print(f"Average F1: {average_f1} Average AUROC: {average_auroc} Average PR_Auc: {average_prAuc}")
    print(f"Standard Deviation F1: {std_deviation_f1} Standard Deviation AUROC: {std_deviation_auroc} Standard Deviation PR_Auc: {std_deviation_prAuc}")
    
    #Get Ensemble results
    print("Getting Ensemble Results")
    fold_folder_path = os.path.join('checkpoints',f"{args['dataset']}/{args['architecture']}/t_{args['transforms']}_m_{multimodality}_cr_{cropping}_o_{oversample}")
    checkpoint_name = f"{model.get_name()}.ckpt"

    inference = Kiadeku_Inference()
    preds,true_labels = inference.get_fold_predictions(fold_folder_path,checkpoint_name,dm)
    ensembled_preds, true_labels = inference.average_predictions(preds, true_labels,mode=mode)
    ensembled_metrics = inference.get_metrices(ensembled_preds, true_labels, dm.get_class_num(),mode=mode)
    print(f"Ensembled Metrics: {ensembled_metrics}")
    
    data = {'average_f1_score':average_f1,'std_f1':std_deviation_f1,'average_auroc_score':average_auroc,'std_auroc':std_deviation_auroc,'average_prAuc_score':average_prAuc,'std_prAuc':std_deviation_prAuc,
    'ensembler_f1':ensembled_metrics['f1'],'ensembler_auroc':ensembled_metrics['auc'],'ensembler_prAuc':ensembled_metrics['prAuc']}

    fold_data ={'fold_f1_scores':test_f1_scores,'fold_auroc_scores':test_auroc_scores,'fold_prAuc_scores':test_prAuc_scores}
    df = pd.DataFrame([data])
    res_path = os.path.join('results',f"{args['dataset']}/{args['architecture']}/t_{args['transforms']}_m_{multimodality}_cr_{cropping}_o_{oversample}")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    df.to_csv(os.path.join(res_path,'result.csv'), index=False)
    pd.DataFrame([fold_data]).to_csv(os.path.join(res_path,'fold_results.csv'), index=False)

if __name__ == "__main__":
    main()