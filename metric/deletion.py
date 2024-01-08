import argparse
import os
import shutil
import time
import random
import pandas as pd

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
import numpy as np
import cv2
import math
from torchvision.utils import draw_bounding_boxes
from sklearn.metrics import roc_auc_score

from datasets import NIHBboxDataset

import wandb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='NIH' , help='Dataor Integral Object Attention githubor Integral Object Attention githubset to train')
parser.add_argument('--plus', default = False, type=str, 
                    help='whether apply icasc++')
parser.add_argument('--ngpu', default=1, type=int, metavar='G',
                    help='number of gpus to use')
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
parser.add_argument('--wandb_key', default='c07987db95186aade1f1dd62754c86b4b6db5af6', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='hphp', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='231212_bbox', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='ICASC++', type=str, help='your wandb project name (you have to change)')


best_prec1 = 0

global result_dir
global iou
global bbox_cnt
global bbox_imgs
global bbox_masks
global bbox_grads

bbox_imgs = []
bbox_masks = []
bbox_grads = []
bbox_cnt = 0
iou = 0

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)

    transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    # normalize
                                    ])
    
    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset)
    # create model
    model = sfocus18(args.dataset, num_classes, pretrained=False, plus=args.plus)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    model.load_state_dict(torch.load('History/NIH/NIH_plain.pth'))

    cudnn.benchmark = True

    # Data loading code
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    wandb.watch(model, log='all', log_freq=10)
    validate(val_loader, model)

def create_binary_mask(heatmap, threshold=0.7):
    # Grad-CAM 히트맵을 이진 마스크로 변환
    binary_mask = torch.where(heatmap >= threshold, 1, 0)
    return binary_mask


def create_background(torch_img, grad_cam_map):
    grad_cam_map = grad_cam_map[0].unsqueeze(dim=0).unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data # (1, 1, W, H), min-max scaling


    #grad_cam_map = grad_cam_map.squeeze() # : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
    b, g, r = grad_heatmap.split(1)
    grad_heatmap = torch.cat([grad_cam_map[0], grad_cam_map[0], grad_cam_map[0]]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.
    mask = 1 - create_binary_mask(grad_heatmap)
    result = torch_img * mask

    return result


def get_hscore(true,false):
    #print(true.min(), true.max())
    true = (true - true.min()) / (true.max() - true.min() + 0.0000001)
    false = (false - false.min()) / (false.max() - false.min() + 0.0000001)
    #print((torch.abs(2 * true - false) - false))
    h_score = ((torch.abs(2 * true - false) - false) / (2*150*150) * 100).sum().item()
    if math.isnan(h_score):
        h_score = 0
    return h_score

def validate(val_loader, model):
    batch_time = AverageMeter()

    global iou
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

    for i, (name, inputs, target) in enumerate(val_loader):
        
        target = target.cuda()
        inputs = inputs.cuda()

        if args.plus == False:
            output, l1, l2, l3, hmap_t, hmaps_f = model(inputs, target)
        else:
            output, l1, l2, l3, hmap_t, hmaps_f, bw, h = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw
        
        temp = []
        for j in range(len(hmap_t)):
            temp.append(create_background(inputs[j], hmap_t[j]))
        
        background_imgs = torch.stack(temp, dim=0)

        if args.plus == False:
            output, l1, l2, l3, hmap_t, hmaps_f = model(background_imgs, target)
        else:
            output, l1, l2, l3, hmap_t, hmaps_f, bw, h = model(background_imgs, target)

    # measure elapsed time
    auc = roc_auc_score(gt, probs)
    print("Test AUC after deletion: {}". format(auc))
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
