import numpy as np
import os
from data import PAIRDataset, ClfDataset
from models import MAX_model
from torch import nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.models import resnet50
from sklearn.metrics import classification_report
from torchvision.ops import sigmoid_focal_loss as criterion
import random
import matplotlib.pyplot as plt
from calc_metrics2 import calc_froc, get_error_boxes, save_cases, join_cases, calc_accuracy
from test import test

def read_data(folder_path):
    pred_list = []
    images = glob.glob(folder_path+"/*.png")
    for j,image_path in enumerate(images):
        item_info = {}
        image_view = "MLO" if ("MLO" in image_path) else "CC"
        preds_path = image_path[:-4]+"_preds.txt"
        preds = torch.tensor(np.loadtxt(preds_path))
        if(preds.shape[0]==0):
            preds = torch.tensor(np.array([[0,0,0,0,-1]]).astype(np.float32))
        if(len(preds.shape)==1):
            preds = preds.unsqueeze(0)
        output = {"boxes": preds[:,:4], 
                "scores": preds[:,4],
                "labels": torch.zeros(preds.shape[0])}
        target_path = image_path[:-4]+".txt"
        if(os.path.isfile(target_path)):
            targets = torch.tensor(np.loadtxt(target_path))
            if(targets.shape[0]!=0):
                if (len(targets.shape)==1): targets = targets.unsqueeze(0)
                targets=targets[:,1:] 
            else:
                targets=torch.tensor([])
        else:
            targets = torch.tensor([])
        item_info['pred'] = output
        item_info['target'] = {"boxes":targets}
        item_info["view"] = image_view
        item_info["img_path"] = image_path
        pred_list.append(item_info)
    return pred_list



def create_plot(data, exp_name):
    # Plot the losses
    loss_names = ['0.05', '0.1', '0.15', '0.2', '0.3', '0.5']
    epochs = list(data.keys())
    epochs = sorted(epochs)
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.grid(True)

    for i, loss_name in enumerate(loss_names):
        loss_values = [data[epoch][i] for epoch in epochs]
        plt.plot(epochs, loss_values, label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity')
    plt.title('Metrics')
    plt.legend()
    plt.savefig('{}/metric_plot.png'.format(exp_name))
    return





def val(dataloader, model, cosine_model=False):
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        pred_list = []
        loss_item = 0
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")

            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            # cc_data[1][:,4] = torch.softmax(cc_data[1][:,4], dim=0)
            # mlo_data[1][:,4] = torch.softmax(mlo_data[1][:,4], dim=0)

            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss_fn(preds, targets)

            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            loss_item += (loss/(mlo_data[0].shape[0]*cc_data[0].shape[0])).item()
        return loss_item


def get_dataloaders(data_file_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = PAIRDataset(pairs_path=data_file_path, transform=transform)
    batch_size = 1
    
    num_samples = len(dataset)
    split_ratio = 0.9
    split_idx = int(split_ratio * num_samples)
    indices = torch.randperm(num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16)

    return train_loader, val_loader

def get_dataloaders2(data_file_path, val_data_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = PAIRDataset(pairs_path=data_file_path, transform=transform)
    val_dataset = PAIRDataset(pairs_path=val_data_path, transform=transform)
    batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16)

    return train_loader, val_loader


def train(data_path, val_data_path, args, exp_name):
    epochs_, lr_, _, resnet_path = args
    train_loader, val_loader = get_dataloaders2(data_path, val_data_path)
    
    model = MAX_model(weights = resnet_path).to("cuda")
    
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer = optim.Adam(model.parameters(), lr=lr_)
    best_loss = 9999
    train_loss = []
    val_loss = []
    epoch_vals = []

    for i in range(epochs_):
        loss_item = 0
        loss = 0
        epoch_vals.append(i+1)
        for j, (mlo_data, cc_data) in enumerate(tqdm(train_loader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")

            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            loss = 0
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            loss = loss/(mlo_data[0].shape[0]*cc_data[0].shape[0])
            loss.backward()
            optimizer.step()            
            loss_item+=loss.item()
        
        val_loss_item = val(val_loader, model)
        print("Epoch:", "{}/{}".format(i+1, epochs_), f"Train Loss: {loss_item:.4f}", f"Val Loss: {val_loss_item:.4f}")

        train_loss.append(loss_item)
        val_loss.append(val_loss_item)

        if(val_loss_item<best_loss):
            best_loss = val_loss_item
            PATH = "{}/{}_epoch.pth".format(exp_name, i)
            torch.save(model.state_dict(), PATH)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    # Plotting training loss
    ax1.plot(epoch_vals, train_loss, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    # Plotting validation loss
    ax2.plot(epoch_vals, val_loss, 'r-', label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()

    # Adjusting the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.savefig('{}/loss_plot.png'.format(exp_name))


    


if __name__ == '__main__':
    dataset_name = "AIIMS"
    resolution = "4k"
    
    train_data_path = "TRAIN_DATA_PATH"
    test_data_path= "TEST_DATA_PATH"
    
    # epochs, lr, scheduler
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    lr = 0.000001
    resnet_path = "./model_weights/resnet_{}.pth".format(dataset_name)
    args = [100, lr, 1, resnet_path]

    exp_name = "./expts_weights/max_exps/{}_{}_resnet/max_{}".format(dataset_name, resolution, lr)

    os.makedirs(exp_name,exist_ok=True)
        
    train(train_data_path, test_data_path, args, exp_name)
    model_path = ""
    tpr, fpr, fpi, sens, precs, pred_list = test(test_data_path, model_path, resnet_path=resnet_path)



