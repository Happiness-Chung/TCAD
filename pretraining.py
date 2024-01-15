import argparse
import os
import shutil
import time
import random
import warnings

import torch
import math
import numpy as np
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
from sklearn.metrics import roc_auc_score
from models.sfocus import sfocus18
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import cv2
from datasets import NIHBboxDataset
from torchvision.utils import draw_bounding_boxes
import pandas as pd

warnings.filterwarnings("ignore")

# Stella added
import wandb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='NIH' , help='ImageNet, CheXpert, NIH, MIMIC')
parser.add_argument('--plus', default=True, type=str, 
                    help='(1) whether apply icasc++')
parser.add_argument('--mask', default= False, type=str, 
                    help='(2) whether apply icasc++')
parser.add_argument('--ngpu', default=1, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default= 10, type=int, metavar='N',
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
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)') # 설명상 default가 None이라서 그렇게 바꿨습니다
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')
best_prec1 = 0

# Stella added
parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='c07987db95186aade1f1dd62754c86b4b6db5af6', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='hphp', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='Pretraining Strategy', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='ICASC++', type=str, help='your wandb project name (you have to change)')


global result_dir
global probs 
global gt    
global k

global bbox_cnt
global bbox_imgs
global bbox_masks
global bbox_grads
global ci
global bbox_df

bbox_imgs = []
bbox_masks = []
bbox_grads = []
bbox_cnt = 0
iou = 0
ci = 0
bbox_df = pd.read_csv('C:/Users/hamdo/Desktop/data/NIH/BBox_List_2017.csv')

def main():

    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)
    base_path = args.base_path
    os.environ["WANDB_API_KEY"] = args.wandb_key
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)

    now = datetime.now()
    result_dir = os.path.join(base_path, "{}_{}H_{}M_P".format(now.date(), str(now.hour), str(now.minute)))
    os.makedirs(result_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    model = sfocus18(args.dataset, 14, pretrained=False, plus=args.plus)
    # define loss function (criterion) and optimizer
    if args.dataset == 'ImageNet':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.dataset == 'CheXpert' or args.dataset == 'MIMIC' or args.dataset == 'NIH':
        criterion = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, 0.1)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    if args.mask == True:
        mask_model = sfocus18(14, pretrained=False, plus=args.plus)
        mask_model = torch.nn.DataParallel(mask_model, device_ids=list(range(args.ngpu)))
        mask_model = mask_model.cuda()
        # optionally resume from a checkpoint
        mask_model.load_state_dict(torch.load(os.path.join(base_path, 'History/2024-01-04_9H/model.pth')))


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(base_path, '2024-01-11_8H/model.pth')))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
    train_loader = torch.utils.data.DataLoader(NIHBboxDataset('C:/Users/hamdo/Desktop/data/NIH', transform=transform), batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    wandb.watch(model, log='all', log_freq=10)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        if args.mask == True:
            train(train_loader, model, criterion, optimizer, epoch, result_dir, mask_model)
        else:
            train(train_loader, model, criterion, optimizer, epoch, result_dir)
     
        scheduler.step()
        ## wandb updates its log per single epoch ##
        

        torch.save(model.state_dict(), result_dir + "/model.pth" )

def calculate_CI(binary_mask, x, y, h, w):

    # BBOX를 이진 마스크로 변환
    bbox_mask = torch.ones((150, 150), dtype=torch.float32,requires_grad=True).cuda()
    bbox_mask2 = torch.zeros((150, 150), dtype=torch.float32,requires_grad=True).cuda()
    x = x*150/1024
    y = y*150/1024
    h = h*150/1024
    w = w*150/1024
    bbox_mask[round(x):round(x+w), round(y):round(y+h)] = 0
    bbox_mask2[round(x):round(x+w), round(y):round(y+h)] = 1

    # 교차 영역과 합집합 영역 계산
    intersection = (binary_mask * bbox_mask).sum()
    intersection2 = (binary_mask * bbox_mask2).sum()
    union = bbox_mask.sum()

    # IoU 계산
    ci = (((1 - intersection2) + intersection) / union)

    return ci

def create_binary_mask(heatmap, threshold=0.7):
    # Grad-CAM 히트맵을 이진 마스크로 변환
    binary_mask = heatmap >= threshold
    return binary_mask

def get_ci(name, grad_cam_map):

    global bbox_df
    sigmoid = nn.Sigmoid()

    grad_cam_map = grad_cam_map[0].unsqueeze(dim=0).unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False) # (1, 1, W, H)'
    
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min) / (map_max - map_min + 1e-7) # (1, 1, W, H), min-max scaling
    for_bbox = grad_cam_map[0]

    x = bbox_df[bbox_df['Image Index'] == name]['Bbox [x'].values[0]
    y = bbox_df[bbox_df['Image Index'] == name]['y'].values[0]
    w = bbox_df[bbox_df['Image Index'] == name]['w'].values[0]
    h = bbox_df[bbox_df['Image Index'] == name]['h]'].values[0]

    # binary_mask = create_binary_mask(for_bbox)
    ci = calculate_CI(for_bbox, x, y, h, w)
    
    return ci

def train(train_loader, model, criterion, optimizer, epoch, dir, mask_model = None):

    global result_dir
    global probs 
    global gt
    global k

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    class_num = 14

    train_loader_examples_num = len(train_loader.dataset)
    probs = np.zeros((train_loader_examples_num, class_num), dtype = np.float32)
    gt = np.zeros((train_loader_examples_num, class_num), dtype = np.float32)
    k = 0
    d_score = 0
    cnt = 0
    total_f = 0

    # switch to train mode
    model.train()

    end = time.time()
    for param in model.parameters():
        param.requires_grad = True
    for i, (name, inputs, target) in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()

        cnt += 1
        
        # compute output
        if args.plus == False:
            output, l1, l2, l3, hmaps_t, hmaps_f = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3
        else:
            output, l1, l2, l3, hmaps_t, hmaps_f, f, d = model(inputs, target)
            loss = criterion(output, target)

        temp = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda() # ㄱㅊ
        for j in range(len(inputs)):
            # temp += ((1 - get_ci(name[j], hmaps_t[j])) + get_ci(name[j], hmaps_f[j]))
            temp += ((get_ci(name[j], hmaps_t[j])))


        temp /= len(inputs)
        loss = temp  
        # loss.requires_grad_(True)
        # loss.retain_grad()
        # sigmoid = nn.Sigmoid()
        # loss = 1 - sigmoid(hmaps_t[0].sum())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        wandb.log({"Functionality loss": total_f / cnt,
                   "CI loss": loss,
                   "D-score": d_score / cnt})

        
        # compute gradient and do SGD step
        # print(loss.is_leaf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # print(torch.equal(a.data, b.data))
        # print("Parameters updated:", any(p.grad is not None for p in model.parameters()))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    if args.dataset == 'ImageNet':
        wandb.log({
        "Epoch":epoch,
        "Train loss":losses.avg,
        "Train Top 1 ACC":top1.avg,
        "Train Top 5 ACC":top5.avg,
    }) 
    elif args.dataset == 'CheXpert' or args.dataset == 'NIH': 
        auc = roc_auc_score(gt, probs)
        print("Training AUC: {}". format(auc))
        wandb.log({
        "Epoch":epoch,
        "Train loss":losses.avg,
        "AUC":auc,
    })   
    
    
def get_hscore(true,false):
    #print(true.min(), true.max())
    true = (true - true.min()) / (true.max() - true.min() + 0.0000001)
    false = (false - false.min()) / (false.max() - false.min() + 0.0000001)
    #print((torch.abs(2 * true - false) - false))
    d_score = ((torch.abs(2 * true - false) - false) / (2*150*150) * 100).sum()
    # .sum().item()
    # if math.isnan(d_score):
    #     d_score = 0
    return d_score

def get_mask(grad_cam_map, idx):

    grad_cam_map = grad_cam_map[idx].unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(32, 32), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, W, H), min-max scaling
    grad_cam_map = torch.cat([grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0)])

    return grad_cam_map

def vis_heatmaps(hmaps, inputs, unnorm, epoch, path):
    f_shape = hmaps[0].shape
    i_shape = inputs[0].shape
    img_tensors = []
    for idx, image in enumerate(inputs):
        hmap = hmaps[idx]
        if f_shape[0] == 1:
            hmap = torch.cat((hmap, torch.zeros(2, f_shape[1], f_shape[2])))
        hmap = (transforms.ToPILImage()(hmap)).resize((i_shape[1], i_shape[2]))
        pil_image = transforms.ToPILImage()(torch.clamp(unnorm(image), 0, 1))
        res = Image.blend(pil_image, hmap, 0.5)
        img_tensors.append(transforms.ToTensor()(res))
    save_image(img_tensors, '{}/{}.png'.format(path, epoch), nrow=8)

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
