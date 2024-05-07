import argparse
import os
import shutil
import time
import random
import pandas as pd
import matplotlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from data_utils import get_datasets
from models.sfocus import sfocus18
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms.functional as Fn
import numpy as np
import cv2
import math
from torchvision.utils import draw_bounding_boxes
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw
from datasets import NIHBboxDataset
import matplotlib.pyplot as plt

import wandb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='NIH' , help='Dataor Integral Object Attention githubor Integral Object Attention githubset to train')
parser.add_argument('--plus', default = True, type=bool, help='whether apply icasc++')
parser.add_argument('--depth', default=256, type=int, metavar='D', help='number of channels of last convolutional block')
parser.add_argument('--ngpu', default=1, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', type=int, default=[50,100], nargs='+', help='LR decay milestones')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')

parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='stellasybae', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='densenet', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='TrustCAD', type=str, help='your wandb project name (you have to change)')


best_prec1 = 0

global result_dir
global bd
global bbox_cnt
global bbox_imgs
global bbox_masks
global bbox_grads

bbox_imgs = []
bbox_masks = []
bbox_grads = []
bbox_cnt = 0
bd = 0

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
                                    transforms.Resize([300,300]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
    transform_raw = transforms.Compose([
                                    transforms.Resize([300,300]),
                                    transforms.ToTensor(),
                                    # normalize
                                    ])
    num_classes = 14

    # create model
    model = sfocus18(args.dataset, "ResNet", num_classes, args.depth, pretrained=False, plus = args.plus)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    model.load_state_dict(torch.load('History/NIH/ECNN256.pth'))

    cudnn.benchmark = True

    # Data loading code
    bbox_loader = torch.utils.data.DataLoader(NIHBboxDataset('C:/Users/hb/Desktop/data/NIH', transform=transform, transform_raw=transform_raw), batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # wandb.watch(model, log='all', log_freq=10)
    validate(bbox_loader, model)

def create_binary_mask(heatmap, threshold=0.5):
    # Grad-CAM 히트맵을 이진 마스크로 변환
    binary_mask = torch.where(heatmap >= threshold, 1, 0)
    return binary_mask

def calculate_BD(grad_t, grad_f, x, y, h, w):

    global bbox_masks

    # BBOX를 이진 마스크로 변환
    bbox_mask = np.zeros_like(grad_t.cpu()) # (3, 300, 300)
    # print(bbox_mask.shape)
    bbox_mask[:,round(x):round(x+w), round(y):round(y+h)] = 1

    # Filtering
    bbox_t = (grad_t.cpu() * bbox_mask)
    bbox_f = (grad_f.cpu() * bbox_mask)
    union = bbox_mask.sum()

    # Bounding-box Difference
    l1 = (bbox_t - bbox_f)
    bd = l1.sum() / union

    return bd

def save_cam(name, target, torch_img, grad_cam_map_t, grad_cam_map_f, index, args, conf = False):

    all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

    args = parser.parse_args()
    if args.dataset == 'NIH':
        bbox_df = pd.read_csv('C:/Users/hb/Desktop/Data/NIH/BBox_List_2017.csv')

    # True map
    grad_cam_map = grad_cam_map_t[0].unsqueeze(dim=0).unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(300, 300), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data # (1, 1, W, H), min-max scaling

    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
    b, g, r = grad_heatmap.split(1)
    grad_result_t = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.
    # grad_result_t = torch.cat([grad_cam_map[0], grad_cam_map[0], grad_cam_map[0]])

    grad_result = grad_result_t.cpu() + torch_img.cpu() # (1, 3, W, H)
    img_t = grad_result.div(grad_result.max()).squeeze() # (3, W, H)

    # False map
    grad_cam_map = grad_cam_map_f[0].unsqueeze(dim=0).unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(300, 300), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data # (1, 1, W, H), min-max scaling

    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
    b, g, r = grad_heatmap.split(1)
    grad_result_f = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.
    # grad_result_f = torch.cat([grad_cam_map[0], grad_cam_map[0], grad_cam_map[0]])

    grad_result = grad_result_f.cpu() + torch_img.cpu() # (1, 3, W, H)
    img_f = grad_result.div(grad_result.max()).squeeze() # (3, W, H)

    # dimension reduction
    # temp = torch.mean(for_bbox, dim=0).squeeze()

    if args.dataset == 'NIH':
        if name in bbox_df['Image Index'].values.tolist():
            # print(name)

            global bd
            global bbox_cnt
            global bbox_imgs
            global bbox_masks
            global bbox_grads 

            x = bbox_df[bbox_df['Image Index'] == name]['Bbox [x'].values[0]
            y = bbox_df[bbox_df['Image Index'] == name]['y'].values[0]
            w = bbox_df[bbox_df['Image Index'] == name]['w'].values[0]
            h = bbox_df[bbox_df['Image Index'] == name]['h]'].values[0]
            
            x = x*300/1024
            y = y*300/1024
            h = h*300/1024
            w = w*300/1024

            # bbox_t, bbox_f = create_binary_mask(grad_result_t, grad_result_f)
            bd += calculate_BD(grad_result_t, grad_result_f, x, y, h, w)
            bbox_cnt += 1

            result_t = Fn.to_pil_image(img_t)
            result_f = Fn.to_pil_image(img_f)
            torch_img = Fn.to_pil_image(torch_img)

            draw = ImageDraw.Draw(torch_img)
            draw.rectangle((x,y,x+w,y+h), outline=(255,0,0), width = 3)
            draw = ImageDraw.Draw(result_t)
            draw.rectangle((x,y,x+w,y+h), outline=(255,0,0), width = 3)
            draw = ImageDraw.Draw(result_f)
            draw.rectangle((x,y,x+w,y+h), outline=(255,0,0), width = 3)

            title= ''
            for l in range(len(target)):
                if target[l] == 1:
                    title += all_classes[l] + ', '
            title = title[:-2]

            
            # matplotlib.rcParams['font.family'] = 'Times New Roman'
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # axes = axes.ravel()
            # axes[0].imshow(torch_img)
            # axes[0].set_title("Input", fontsize=16)
            # axes[0].axis('off')
            # axes[1].imshow(result_t)
            # axes[1].set_title("True", fontsize=16)
            # axes[1].axis('off')
            # axes[2].imshow(result_f)
            # axes[2].set_title("False", fontsize=16)
            # axes[2].axis('off')
            
            # plt.suptitle(title, fontsize=20)
            # plt.savefig(os.getcwd() + '/Results/bbox/result{}.jpg'.format(index))
            # plt.close()


    

def validate(val_loader, model):
    batch_time = AverageMeter()
    bbox_df = pd.read_csv('C:/Users/hb/Desktop/Data/NIH/BBox_List_2017.csv')

    global ci
    global bbox_cnt
    global bbox_imgs
    global bbox_grads

    global probs 
    global gt
    global k

    # switch to evaluate mode
    # model.eval()
    end = time.time()
    h_score = 0
    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    if args.dataset == 'CheXpert':
        class_num = 10
    elif args.dataset == 'NIH':
        class_num = 14

    val_loader_examples_num = len(val_loader.dataset)
    probs = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    gt = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    k = 0

    for i, (name, raw_inputs, inputs, target) in enumerate(val_loader):
        
        target = target.cuda()
        inputs = inputs.cuda()
        raw_inputs = raw_inputs.cuda()

        # compute output
        if args.plus == False:
            output, l1, l2, l3, hmap_t, hmaps_f = model(inputs, target)
        else:
            output, l1, l2, l3, hmap_t, hmaps_f, bw, h = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw
        
        bbox_imgs = []
        bbox_grads = []
        for j in range(len(hmap_t)):
            if name[j] in bbox_df['Image Index'].values.tolist():
                save_cam(name[j], target[j], raw_inputs[j], hmap_t[j], hmaps_f[j],  i*len(inputs) + j, args)
        # wandb.log({"Final Images":bbox_imgs,"Grads":bbox_grads})

    # measure elapsed time
    print("Bounding-box difference : ", bd / bbox_cnt)
    # wandb.log({"Final Images":bbox_imgs})
    batch_time.update(time.time() - end)
    end = time.time()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dataset, output, target, topk=(1,)):
    
    """Computes the precision@k for the specified values of k"""
    sigmoid = torch.nn.Sigmoid()
    res = []
    global probs 
    global gt
    global k
    
    if dataset == 'ImageNet':
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    
    elif dataset == 'CheXpert' or dataset == 'NIH':
        
        # For AUC ROC
        probs[k: k + output.shape[0], :] = output.cpu()
        gt[   k: k + output.shape[0], :] = target.cpu()
        k += output.shape[0] 
        
        # For accuracy
        preds = np.round(sigmoid(output).cpu().detach().numpy())
        targets = target.cpu().detach().numpy()
        test_sample_number = len(targets)* len(output[0])
        test_correct = (preds == targets).sum()
        
        res.append([test_correct / test_sample_number * 100])
        res.append([0])
    
    return res

if __name__ == '__main__':

    main()
