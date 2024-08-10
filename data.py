import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import os
import glob
from torchvision import transforms
from tqdm import tqdm

class ClfDataset(Dataset):
    def __init__(self, pairs_path, transform=None):
        self.transform = transform
        self.images = []
        self.targets = []
        self.pair_data = []
        for i,folder in enumerate(tqdm(os.listdir(pairs_path))):
            folder_path = os.path.join(pairs_path, folder)
            data_pt = {"folder_id": folder_path}
            data_pt["mal"] = 1 if len(os.listdir(folder_path))==6 else 0
            image_files = glob.glob(folder_path+"/*.png")  
            # import pdb; pdb.set_trace()          
            for j,image_file in enumerate(image_files):
                if("MLO" in image_file or "ML" in image_file):
                    data_pt["MLO_img"] = image_file
                    data_pt["MLO_preds"] = image_file[:-4]+"_preds.txt"
                    data_pt["MLO_gt"] = image_file[:-4]+".txt" if(data_pt["mal"]==1) else None
                    image_crops, crop_targets = self.get_images_targets(data_pt["MLO_img"], data_pt["MLO_gt"], data_pt["MLO_preds"])
                    self.images+=image_crops
                    self.targets+=crop_targets
                elif("CC" in image_file):
                    data_pt["CC_img"] = image_file
                    data_pt["CC_preds"] = image_file[:-4]+"_preds.txt"
                    data_pt["CC_gt"] = image_file[:-4]+".txt" if(data_pt["mal"]==1) else None
                    image_crops, crop_targets = self.get_images_targets(data_pt["CC_img"], data_pt["CC_gt"], data_pt["CC_preds"])
                    self.images+=image_crops
                    self.targets+=crop_targets
                else:
                    print("ERROR while loading datastet", image_file)
                    exit(0)
            if("MLO_img" not in data_pt.keys() or "CC_img" not in data_pt.keys() ):
                print("Both view are not present in a folder", folder_path)
            self.pair_data.append(data_pt)
        # import pdb; pdb.set_trace() 


    def yolo_to_pillow(self, box, image_size):
        x, y, w, h = box
        img_w, img_h= image_size
        left = int((x - w/2) * img_w)
        top = int((y - h/2) * img_h)
        right = int((x + w/2) * img_w)
        bottom = int((y + h/2) * img_h)
        return (left, top, right, bottom)   

    def get_images_targets(self, image_file, gt_file, pred_file, topk = None):
        image = Image.open(image_file).convert("RGB")
        crops = []
        targets = []
        proposal_data = np.loadtxt(pred_file).astype(np.float32)[:topk]
        if(len(proposal_data.shape)==1):
            proposal_data = np.expand_dims(proposal_data, 0)
        gt_data = None
        if(gt_file!=None):
            gt_data = np.loadtxt(gt_file).astype(np.float32)
        for i, proposal in enumerate(proposal_data):
            box_pillow = self.yolo_to_pillow(proposal[:4], image.size)
            crop0 = image.crop(box_pillow)
            targets.append(self.true_positive(gt_data, proposal[:4]))
            crops.append(crop0)        
        return crops, targets
    

    def true_positive(self, gts, pred):
        if(type(gts)==type(None)):
            return 0
        # import pdb; pdb.set_trace()
        gts = np.expand_dims(gts, axis=0) if(len(gts.shape)==1) else gts
        gts = gts[:,1:]
        for i,gt in enumerate(gts):
            # If center of pred is inside the gt, it is a true positive
            box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
            if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                    pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
                return 1
        return 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        img = self.transform(img)
        
        return img, target


class PAIRDataset(Dataset):
    def __init__(self, pairs_path, transform=None):
        self.pair_data = []
        self.transform = transform
        for i,folder in enumerate(os.listdir(pairs_path)):
            folder_path = os.path.join(pairs_path, folder)
            data_pt = {"folder_id": folder_path}
            data_pt["mal"] = 1 if len(os.listdir(folder_path))==6 else 0
            image_files = glob.glob(folder_path+"/*.png")            
            for j,image_file in enumerate(image_files):
                if("MLO" in image_file or "ML" in image_file):
                    data_pt["MLO_img"] = image_file
                    data_pt["MLO_preds"] = image_file[:-4]+"_preds.txt"
                    data_pt["MLO_gt"] = image_file[:-4]+".txt" if(data_pt["mal"]==1) else None
                elif("CC" in image_file):
                    data_pt["CC_img"] = image_file
                    data_pt["CC_preds"] = image_file[:-4]+"_preds.txt"
                    data_pt["CC_gt"] = image_file[:-4]+".txt" if(data_pt["mal"]==1) else None
                else:
                    print("ERROR while loading datastet", image_file)
                    exit(0)
            if("MLO_img" not in data_pt.keys() or "CC_img" not in data_pt.keys() ):
                print("Both view are not present in a folder", folder_path)
            self.pair_data.append(data_pt)
        

    def __len__(self):
        return len(self.pair_data)

    def yolo_to_pillow(self, box, image_size):
        x, y, w, h = box
        img_w, img_h= image_size
        left = int((x - w/2) * img_w)
        top = int((y - h/2) * img_h)
        right = int((x + w/2) * img_w)
        bottom = int((y + h/2) * img_h)
        return (left, top, right, bottom)

    def save_points(self, image, proposals, crops):
        # import pdb; pdb.set_trace()
        draw = ImageDraw.Draw(image)
        for j,box in enumerate(proposals):
            box = self.yolo_to_pillow(box[:4], image.size)
            bbox = ((box[0], box[1]), (box[2], box[3]))
            draw.rectangle(bbox, outline ="red")
        
        img_name = len(os.listdir("check_data")) 
        image.save("check_data/{}.png".format(img_name))
        for i,img in enumerate(crops):
            img.save("check_data/{}_{}.png".format(img_name, i))
        exit(0)


    def true_positive(self, gts, pred):
        if(type(gts)==type(None)):
            return 0
        gts = np.expand_dims(gts, axis=0) if(len(gts.shape)==1) else gts
        gts = gts[:,1:]
        for i,gt in enumerate(gts):
            # If center of pred is inside the gt, it is a true positive
            box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
            if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                    pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
                return 1
        return 0

    def get_view_data(self, data_pt, view = "MLO", topk = None):
        image = Image.open(data_pt["{}_img".format(view)]).convert("RGB")
        proposal_data = np.loadtxt(data_pt["{}_preds".format(view)]).astype(np.float32)[:topk]
        if(proposal_data.shape[0]==0):
            proposal_data = np.array([[0,0,0,0,-1]]).astype(np.float32)
        elif(len(proposal_data.shape)==1):
            proposal_data = np.expand_dims(proposal_data, 0)
        gt_data = None
        if(data_pt["{}_gt".format(view)]!=None):
            gt_data = np.loadtxt(data_pt["{}_gt".format(view)]).astype(np.float32)
        image_crops = []
        pil_crops = []
        targets = []
        for i, proposal in enumerate(proposal_data):
            box_pillow = self.yolo_to_pillow(proposal[:4], image.size)
            crop0 = image.crop(box_pillow)
            crop = self.transform(crop0)
            image_crops.append(crop.unsqueeze(0))
            targets.append(self.true_positive(gt_data, proposal[:4]))
            pil_crops.append(crop0)
        targets = np.expand_dims(targets, axis=1).astype(np.float32)
        proposal_data = np.concatenate((proposal_data, targets), axis=1)
        # if(sum(targets)>0):
        #     self.save_points(image, proposal_data, pil_crops)
        return torch.cat(image_crops, axis=0), torch.from_numpy(proposal_data), data_pt["folder_id"] 
        

    def __getitem__(self, idx):
        # Load the image and proposals data for the given index
        data_pt = self.pair_data[idx]
        mlo_data= self.get_view_data(data_pt, "MLO")
        cc_data = self.get_view_data(data_pt, "CC")

        return mlo_data, cc_data
 


if __name__ == '__main__':
    # Example usage:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    DATA_PATH = "DATA_PATH"
    # dataset = PAIRDataset(pairs_path=DATA_PATH, transform=transform)
    dataset = ClfDataset(pairs_path=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (img, target) in enumerate(dataloader):
        import pdb; pdb.set_trace()
        # Do something with the batch of images and proposals
        pass