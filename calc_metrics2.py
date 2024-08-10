
import os, sys
import torch, json
import numpy as np

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image




def get_confmat_clf(pred_list, threshold=0.1):
    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        if(len(gt_data['boxes'])!=0 and len(pred_boxes)!=0):
            out_array[0]+=1
        elif(len(gt_data['boxes'])==0 and len(pred_boxes)!=0):
            out_array[2]+=1
        elif(len(gt_data['boxes'])!=0 and len(pred_boxes)==0):
            out_array[3]+=1
        else:
            out_array[1]+=1
        conf_mat+=out_array
    return conf_mat

def get_confmat(pred_list, threshold = 0.3):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    error_image = np.zeros((len(pred_list)))
    conf_mat_idx = []
    # flag= False
    for i, data_item in enumerate(pred_list):
        # if(data_item["img_path"].split("/")[-2] == "726"):
            # import pdb; pdb.set_trace()
            # flag=True
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        # print(pred, select_mask)
        # import pdb; pdb.set_trace()
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred in pred_boxes:
                if true_positive(gt_box, pred):
                    add_tp = True
                else:
                    new_preds.append(pred)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1
            else:
                out_array[3] += 1
        # if(flag):
        #     pdb.set_trace()
        out_array[2] = len(pred_boxes)
        conf_mat+=out_array
        conf_mat_idx.append(out_array)
        if(out_array[2]!=0 or out_array[3]!=0):
            error_image[i] = 1
    return conf_mat, error_image, conf_mat_idx



    
import matplotlib.pyplot as plt
def save_plot(senses, fps, data="l"):
    plt.plot(fps, senses)
    plt.xlabel("False positives per image")
    plt.ylabel("Sensitivity")
    plt.title("{} FROC".format(data))
    plt.legend()
    plt.grid(True)
    plt.savefig("{}_froc_plot.png".format(data))
    plt.clf()
    
def calculate_auc(confusion_matrices):
    tpr_prev = 0
    fpr_prev = 0
    auc_score = 0
    
    sorted_matrices = sorted(confusion_matrices, key=lambda x: x['threshold'], reverse=True)
    # import pdb; pdb.set_trace()
    tprs = []; fprs = []
    precs = []
    for cm in sorted_matrices:
        tp, tn, fp, fn = cm["conf_mat"]

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        pres = 1 if tp ==0 else tp / (tp + fp)
        
        tprs.append(tpr); fprs.append(fpr); precs.append(pres)
        width = fpr-fpr_prev
        if(width<0):
            import pdb; pdb.set_trace()
        avg_height = (tpr + tpr_prev) / 2
        auc_score += width * avg_height
        tpr_prev = tpr
        fpr_prev = fpr
    return auc_score, tprs, fprs, precs


def calc_accuracy(pred_data, num_thresh=1000):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    metrics = np.zeros((num_thresh, 4))

    #tp, tn, fp, fn
    conf_mat_thresh = []
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat= get_confmat_clf(pred_data, thresh_val)
        conf_mat_thresh.append({"threshold":thresh_val, "conf_mat": conf_mat})
        
        pres = conf_mat[0]/(conf_mat[0]+conf_mat[2]+ 1) + 0.0001
        recall = conf_mat[0]/(conf_mat[0]+conf_mat[3]+ 1) + 0.0001
        metrics[i,0] = 2*pres*recall/(pres+recall)
        metrics[i,1] = (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3])
        metrics[i,2] = pres
        metrics[i,3] = recall
        
        # if(thresh_val>0.028 and thresh_val < 0.032):
        #     print("Threshold:", thresh_val)
        #     print("F1 score:", 2*pres*recall/(pres+recall))
        #     print("Accuracy:", (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3]))
    max_f1, max_acc, max_pres, max_recall = np.argmax(metrics, axis=0)
    auc, tprs, fprs, precs = calculate_auc(conf_mat_thresh)
    print(auc)
    print("Max F1 score and Accuracy:", metrics[max_f1],  "Threshold:", thresholds[max_f1])
    print("F1 score and Max Accuracy:", metrics[max_acc], "Threshold:", thresholds[max_acc])
    
    # print("Acc, F1, Prec, Recall, AUC")
    # print("%.3f %.3f %.3f %.3f %.3f"%(metrics[max_acc][1], metrics[max_acc][0], metrics[max_acc][2], metrics[max_acc][3], auc))
    return tprs, fprs, precs
    
    

def calc_froc(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5, 0.55 ,0.6, 0.63,0.7,0.8, 0.84,1,1.1,1.2,1.5,1.8,1.9,2,2.4,2.7,3,4.4,5.4], num_thresh = 500):
    num_images = len(pred_data)
    # fps_req = np.linspace(0,6,num_thresh)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat,_,_ = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat
    
    sensitivity = np.zeros((num_thresh)) #recall
    specificity = np.zeros((num_thresh)) #presicion
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
        if((conf_mat[0]+conf_mat[2])==0):
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[2])

    senses_req = []
    fpi_calc=[]
    froc_fpis =[]
    froc_sens =[]
    froc_thres =[]
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            # import pdb; pdb.set_trace()
            if f/num_images < fp_req:                
                # import pdb; pdb.set_trace()
                senses_req.append(sensitivity[i-1])
                froc_sens.append(sensitivity[i-1])
                froc_thres.append(thresholds[i])
                froc_fpis.append(conf_mat_thresh[i-1][2]/num_images)
                fpi_calc.append(fp_req)
                print(fp_req, sensitivity[i-1], thresholds[i])
                break
    # save_plot(senses_req, fps_req, data="aiims")
    # print(fps_req)
    # print(senses_req)
    return froc_fpis, froc_sens, froc_thres
    

def save_boxes(img_path, folder_path, boxes, labels, scores=None):

    img = cv2.imread(img_path)
    img_name = img_path.split("/")[-1]
    for i,box in enumerate(boxes):
        # Convert the box coordinates from YOLO format to pixel coordinates
        x = int((box[0]-box[2]/2) * img.shape[1])
        y = int((box[1]-box[3]/2) * img.shape[0])
        w = int(box[2] * img.shape[1])
        h = int(box[3] * img.shape[0])

        # Draw the box on the image
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if(labels[i]=="pred"):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
            if(scores!=None):
                cv2.putText(img, '%.3f' %scores[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
    # print(os.path.join(folder_path, img_name))
    cv2.imwrite(os.path.join(folder_path, img_name), img)



def save_cases(cases_list, pred_list, save_folder, thresh = 0.1041041041041041):
    os.makedirs(save_folder, exist_ok=True)
    items = []
    for i,case in enumerate(cases_list):
        flag = 0
        for k in range(len(pred_list)):
            if(int(pred_list[k]['img_path'].split("/")[-2])==case):
                items.append(pred_list[k])
                flag+=1
        if(flag!=2):
            print("case not found", case)
            return
    
    for i,item in enumerate(items):
        case = int(item['img_path'].split("/")[-2])
        # print(item["img_path"], case)
        # import pdb; pdb.set_trace()
        case_folder = os.path.join(save_folder, str(case))
        select_mask = item["pred"]['scores'] > thresh
        box_labels = ["pred" for item2 in item["pred"]['labels'][select_mask]] + ["gt" for item2 in item["target"]['boxes']]
        boxes = torch.cat((item["pred"]['boxes'][select_mask],item["target"]['boxes']))
        os.makedirs(case_folder, exist_ok=True)
        save_boxes(item["img_path"], case_folder, boxes, box_labels, scores = item["pred"]['scores'][select_mask])


import glob
from PIL import Image

def concatenate_images(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]

    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    new_image = Image.new('RGB', (total_width, max_height))


    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width
        
    new_image.save(output_path)



def join_cases(cases_list, src_path):
    def sort_images(img_list):
        if("CC" in img_list[0]):
            img_list.append(img_list.pop(0))
        return img_list
            
    save_path=os.path.join(src_path, "join")
    os.makedirs(save_path, exist_ok=True)
    images = {}
    for i,case in enumerate(cases_list):
        images_van = glob.glob(os.path.join(src_path,"van",str(case))+"/*.png"); images_van = sort_images(images_van)
        images_attn = glob.glob(os.path.join(src_path,"attn",str(case))+"/*.png"); images_attn = sort_images(images_attn)
        images[case]=images_van+images_attn
    
    for (key,value) in tqdm(images.items()):
        trgt_img_path = os.path.join(save_path, "combined_img_{}.png".format(key))
        concatenate_images(value, trgt_img_path)
        
        
        
        


def get_error_boxes(pred_data, save_folder, thresh = 0.1313131313):
    os.makedirs(save_folder, exist_ok=True)
    _, error_image, conf_mats = get_confmat(pred_data, thresh)
    # error_idxs = np.where(error_image!=0)
    error_idxs = np.where(np.array(conf_mats)[:,3]>=1)
    import pdb; pdb.set_trace()
    selected_pred = np.array(pred_data)[error_idxs[0]]
    selected_conf = np.array(conf_mats)[error_idxs[0]]
    for i, item in enumerate(tqdm(selected_pred)):
        # import pdb; pdb.set_trace()
        folder_path = os.path.join(save_folder, item["img_path"].split("/")[-2])

        select_mask = item["pred"]['scores'] > thresh
        box_labels = ["pred" for item2 in item["pred"]['labels'][select_mask]] + ["gt" for item2 in item["target"]['boxes']]
        boxes = torch.cat((item["pred"]['boxes'][select_mask],item["target"]['boxes']))
        if(len(["gt" for item2 in item["target"]['boxes']])>0):
            os.makedirs(folder_path, exist_ok=True)
            save_boxes(item["img_path"], folder_path, boxes, box_labels, scores = item["pred"]['scores'][select_mask])
        # gt_data = item['target']
        # pred = item['pred']
        # scores = pred['scores']
        # select_mask = scores > thresh
        # box_labels = ["pred" for item in pred['boxes'][select_mask]] + ["gt" for item in gt_data['boxes']]
        # pred_dict = {
        #     'boxes': torch.cat((pred['boxes'][select_mask],gt_data['boxes'])),
        #     'size': gt_data['size'],
        #     'box_label': box_labels,
        #     'image_id': gt_data['image_id']
        # }
        # vslzr.visualize(item['image'], pred_dict, savedir=save_folder, show_in_console=False)

