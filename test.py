import numpy as np
import os
from data import PAIRDataset
from models import MAX_model
from torch import nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from torchvision.models import resnet50
from sklearn.metrics import classification_report
import random
from train import read_data
from calc_metrics2 import calc_froc, get_error_boxes, save_cases, join_cases, calc_accuracy



def change_confs(folder_path, mlo_scores, cc_scores):
    num_props = mlo_scores.shape[0]
    folder_data = read_data(folder_path[0])
    if(folder_data[0]["view"]=="CC"):
        folder_data.append(folder_data.pop(0))

    folder_data[0]["pred"]["scores"] = folder_data[0]["pred"]["scores"][:num_props]
    folder_data[1]["pred"]["scores"] = folder_data[1]["pred"]["scores"][:num_props]
    folder_data[0]["pred"]["boxes"] = folder_data[0]["pred"]["boxes"][:num_props]
    folder_data[1]["pred"]["boxes"] = folder_data[1]["pred"]["boxes"][:num_props]
    folder_data[0]["pred"]["labels"] = folder_data[0]["pred"]["labels"][:num_props]
    folder_data[1]["pred"]["labels"] = folder_data[1]["pred"]["labels"][:num_props]

    # import pdb; pdb.set_trace()
    # folder_data[0]["pred"]["new_scores"] = mlo_scores
    # folder_data[1]["pred"]["new_scores"] = cc_scores
    folder_data[0]["pred"]["scores"] = mlo_scores
    folder_data[1]["pred"]["scores"] = cc_scores
    
    return folder_data


def test(data_file_path, model_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    sigmoid = nn.Sigmoid()
    dataset = PAIRDataset(pairs_path=data_file_path, transform=transform)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16)
    
    model = MAX_model(weights = None).to("cuda")
            
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        pred_list = []
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")
            
            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            preds_mlo = model(mlo_data, cc_data)
            preds_mlo = sigmoid(preds_mlo)
            targets_mlo = mlo_data[1][:,5]

            preds_cc = model(cc_data, mlo_data)
            preds_cc = sigmoid(preds_cc)
            targets_cc= cc_data[1][:,5]

            pred_list+=change_confs(mlo_data[2], preds_mlo.cpu(), preds_cc.cpu())
            # pred_list = None
        fps_req, senses_req, thresh = calc_froc(pred_list)
        # fpi, sens = calc_froc(pred_list)
        fpi, sens = [], []
        tpr, fpr, precs = calc_accuracy(pred_list)
        
    return  tpr, fpr, fpi, sens, precs, pred_list




def save_plot_values(model_name, dataset, data):
    tpr, fpr, fpi, sen, precs = data
    
    auc_target_path = os.path.join("AUC","{}_{}_auc".format(dataset, model_name))
    np.save(auc_target_path, np.array([fpr,tpr]))
    
    froc_target_path = os.path.join("FROC","{}_{}_froc".format(dataset, model_name))
    np.save(froc_target_path, np.array([fpi,sens]))
 
    pr_target_path = os.path.join("PR","{}_{}_pr".format(dataset, model_name))
    np.save(pr_target_path, np.array([precs,tpr]))
    


if __name__ == '__main__':
    model_path = ""
    test_data_path = ""

    print(model_path, test_data_path)
    tpr, fpr, fpi, sens, precs, pred_list = test(test_data_path, model_path)
