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
from models.DenseNet import densenet201, densenet121
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import cv2
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

warnings.filterwarnings("ignore")

# Stella added
import wandb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='CheXpert' , help='ImageNet, CheXpert, NIH, MIMIC')
parser.add_argument('--plus', default = True, type=str, help='(1) whether apply icasc++')
parser.add_argument('--model', default='ResNet', type=str, help='Base model architecture')
parser.add_argument('--depth', default= 192, type=int, metavar='G', help='the number of channels of the last convolutional blocks')
parser.add_argument('--mask', default= False, type=str, help='(2) whether apply icasc++')
parser.add_argument('--ngpu', default= 1, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default= 40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', type=int, default=[50,100], nargs='+', help='LR decay milestones') #####
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default= False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')
best_prec1 = 0

# Stella added
parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='stellasybae', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='densenet', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='TrustCAD', type=str, help='your wandb project name (you have to change)')
parser.add_argument('--layer_depth', default=1, type=int, help='depth of last layer')
parser.add_argument('--seed', default=1, metavar='BS', type=int, help='seed for split file', choices=[1,2,3])
parser.add_argument('--loss_tf', action='store_true', help='true attention map vs. false attention map') 

global result_dir
global probs 
global gt    
global k
global best_validation_score
best_validation_score = 0

def main():

    def build_lrfn(lr_start=0.000002, lr_max=0.00010, 
               lr_min=0, lr_rampup_epochs=8, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

        def lrfn(epoch):
            if epoch < lr_rampup_epochs:
                optimizer.param_groups[0]['lr'] = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
            elif epoch < lr_rampup_epochs + lr_sustain_epochs:
                optimizer.param_groups[0]['lr'] = lr_max
            else:
                optimizer.param_groups[0]['lr'] = (lr_max - lr_min) *\
                    lr_exp_decay**(epoch - lr_rampup_epochs\
                                    - lr_sustain_epochs) + lr_min
            return optimizer.param_groups[0]['lr']
        
        return lrfn

    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)
    base_path = args.base_path
    os.environ["WANDB_API_KEY"] = args.wandb_key
    os.environ["WANDB_MODE"] = args.wandb_mode
    # wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)

    now = datetime.now()
    result_dir = os.path.join(base_path, "{}_{}H".format(now.date(), str(now.hour)))
    os.makedirs(result_dir, exist_ok=True)
    c = open(result_dir + "/config.txt", "w")
    c.write("plus: {}, depth: {}, dataset: {}, epochs: {}, lr: {}, momentum: {},  weight-decay: {}, seed: {}".format(args.plus, str(args.depth), args.dataset, str(args.epochs), str(args.lr), str(args.momentum),str(args.weight_decay), str(args.seed)))
    open(result_dir + "/validation_performance.txt", "w")
    open(result_dir + "/test_performance.txt", "w")


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, test_dataset, num_classes, unorm = get_datasets(args)
    # create model
    kwargs = vars(args)
    model = sfocus18(pretrained=False, **kwargs)
    # model = densenet201()
    # model = densenet121()
    # define loss function (criterion) and optimizer
    if args.dataset == 'ImageNet':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.dataset == 'CheXpert' or args.dataset == 'MIMIC' or args.dataset == 'NIH':
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        criterion2 = AUCMLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                         momentum=args.momentum,
    #                         weight_decay=args.weight_decay)
    # Loss = 
    # optimizer = PESG(model.model.parameters(), criterion, lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.00001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, 0.1)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.mask == True:
        mask_model = sfocus18(args.dataset, num_classes, depth = args.depth, pretrained=False, plus=args.plus)
        mask_model = torch.nn.DataParallel(mask_model, device_ids=list(range(args.ngpu)))
        mask_model = mask_model.cuda()
        # optionally resume from a checkpoint
        mask_model.load_state_dict(torch.load(os.path.join(base_path, 'CheXpert256_floss.pth')))

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(base_path, '2024-03-14_14H/model.pth')))

    cudnn.benchmark = True

    # Data loading code
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(test_loader, model, criterion, unorm, -1, PATH)
        return
    PATH = os.path.join('./checkpoints/SF', args.dataset, args.prefix)
    os.makedirs(PATH, exist_ok=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # print("Train: ", int(0.9 * len(train_loader) * args.batch_size), "Validation: ", len(train_loader) - int(0.9 * len(train_loader) * args.batch_size))
    # train_loader, val_loader = torch.utils.data.random_split(train_loader, [int(0.9 * len(train_loader)), len(train_loader) - int(0.9 * len(train_loader))])
    # wandb.watch(model, log='all', log_freq=10)
    for epoch in range(args.start_epoch, args.epochs):

        if epoch == 0:
            prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, result_dir)
            prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, result_dir)
        
        # train for one epoch
        if args.mask == True:
            train(train_loader, val_loader, test_loader, model, criterion, criterion2, optimizer, epoch, result_dir, mask_model)
        else:
            train(train_loader, val_loader, test_loader, model, criterion, criterion2, optimizer, epoch, result_dir, PATH, unorm)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, result_dir)
        prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, result_dir)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # scheduler.step()
        ## wandb updates its log per single epoch ##
        

def train(train_loader,val_loader, test_loader, model, criterion, criterion2, optimizer, epoch, dir, PATH, unorm, mask_model = None):

    global result_dir
    global probs 
    global gt
    global k

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.dataset == 'CheXpert':
        class_num = 10
    elif args.dataset == 'NIH':
        class_num = 14

    train_loader_examples_num = len(train_loader.dataset)
    probs = np.zeros((train_loader_examples_num, class_num), dtype = np.float32)
    gt = np.zeros((train_loader_examples_num, class_num), dtype = np.float32)
    k = 0
    d_score = 0
    cnt = 0
    total_bw = 0

    sigmoid =  nn.Sigmoid()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()

        cnt += 1

        if args.mask == True:
            output, l1, l2, l3, hmaps_t_original, hmaps_f_original, bw, h = mask_model(inputs, target)
            augmented_inputs = mask_img(inputs, hmaps_t_original) 
            # plt.imshow(augmented_inputs[i].permute(1,2,0).cpu().numpy())
            # plt.savefig('History/aug{}_8'.format(i))     
            # inputs = torch.nan_to_num(inputs)
        
        # inputs = torch.cat([inputs, augmented_inputs], dim=0)
        # target = torch.cat([target, target], dim=0)
        # compute output
        if args.plus == False:
            # ResNet
            # output, l1, l2, l3, hmaps_t, hmaps_f = model(inputs, target)
            # DenseNet
            output, l1, l2, l3, hmap_t, hmaps_f = model(inputs, target)
            # output = model(inputs, parallel_last = False)
            # loss = criterion(output, target)+l1+l2+l3
            loss = criterion(output, target) + criterion2(sigmoid(output),target)
        else:
            output, l1, l2, l3, hmaps_t, hmaps_f, bw, d_main = model(inputs, target)
            # aug_output, _, _, _, hmaps_t_aug, hmaps_f_aug, bw, d = model(augmented_inputs, target)
            if args.mask == True:
                hmaps_t_original = torch.stack(hmaps_t_original, dim = 0)
                hmaps_f_original = torch.stack(hmaps_f_original, dim = 0)
            # hmaps_t = torch.stack(hmaps_t, dim = 0)
            # hmaps_f = torch.stack(hmaps_f, dim = 0)

            total_bw += bw
            # aug_loss1 = torch.mean(torch.abs((output) - (aug_output)))
            # aug_loss2 = hmaps_t_original.sum() / hmaps_t.sum()
            # aug_loss3 = hmaps_f.sum() / hmaps_f_original.sum()

            # loss = criterion(output, target) + aug_loss1 + (aug_loss2 + aug_loss3) * 0.7
            loss = criterion(output, target) + criterion2(sigmoid(output),target) + bw
        
            d_score += d_main

        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # wandb.log({"BW loss": total_bw / cnt,
        #            "Total loss": losses.val,
        #            "H-score": d_score / cnt})

        
        # compute gradient and do SGD step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # scheduler.step()
        
        if i % 2000 == 0 and i != 0:
            tempk = k
            tempprobs = probs
            tempgt = gt
            prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, dir)
            prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, dir)
            k = tempk
            probs = tempprobs
            gt = tempgt
            model.train()

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
            print('H-score: ', d_score / cnt)
    
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
    #     wandb.log({
    #     "Epoch":epoch,
    #     "Train loss":losses.avg,
    #     "AUC":auc,
    # })   
    
def create_binary_mask(heatmap, threshold = 0.6):
    # Grad-CAM 히트맵을 이진 마스크로 변환
    binary_mask = torch.where(heatmap >= threshold, 1, 0.4)
    return binary_mask

def mask_img(imgs, cams):

    augmented_imgs = []

    for i in range(len(imgs)):
        grad_cam_map = cams[i].unsqueeze(dim=0)
        grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False) # (1, 1, W, H)
        map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
        grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data # (1, 1, W, H), min-max scaling

        #grad_cam_map = grad_cam_map.squeeze() # : (224, 224)
        grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
        grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
        b, g, r = grad_heatmap.split(1)
        grad_heatmap = torch.cat([grad_cam_map[0], grad_cam_map[0], grad_cam_map[0]]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.
        mask = create_binary_mask(grad_heatmap)
        result = imgs[i] * mask
        augmented_imgs.append(result)

    augmented_inputs = torch.stack(augmented_imgs, dim=0)
    augmented_inputs = augmented_inputs.cuda()
    
    return augmented_inputs
    
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

def validate(val_loader, model, criterion, unorm, epoch, PATH, dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    global probs 
    global gt
    global k
    global best_validation_score

    sigmoid =  nn.Sigmoid()

    if args.dataset == 'CheXpert':
        class_num = 10
    elif args.dataset == 'NIH':
        class_num = 14

    val_loader_examples_num = len(val_loader.dataset)
    probs = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    gt = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    k = 0
    cnt = 0
    d_score = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        cnt += 1
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            # ResNet
            output, l1, l2, l3, hmap_t, hmaps_f  = model(inputs, target)
            # loss = criterion(output, target)+l1 + l2 + l3
            # DenseNet
            # output = model(inputs, parallel_last = False)
            loss = criterion(output, target)
        else:
            output, l1, l2, l3, hmap_t, hmaps_f, bw, h = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw
            d_score += h

        true_overlays = []
        false_overlays = []

        # for j in range(len(hmap_t)):
        #     true_overlays.append(wandb.Image(save_cam(inputs[j], hmap_t[j], i*len(inputs) + j ), caption="true"))
        #     false_overlays.append(wandb.Image(save_cam(inputs[j], hmaps_f[j], i*len(inputs) + j, conf=True), caption="false"))
        

        # wandb.log({
        #     'AttentionMap/True': true_overlays,
        #     'AttentionMap/False': false_overlays,
        # }) 

        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print("H-score: ", d_score / cnt)

    if args.dataset == 'ImageNet':
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        wandb.log({
            "Valid loss":losses.avg,
            "Valid Top 1 ACC":top1.avg,
            "Valid Top 5 ACC":top5.avg,
        })   
        f = open(dir + "/performance.txt", "a")
        f.write(str(top1.avg.item()) + "\n")
        f.close()
    elif args.dataset == 'CheXpert' or args.dataset == 'NIH': 
        auc = roc_auc_score(gt, probs)
        if auc > best_validation_score:
            best_validation_score = auc
            torch.save(model.state_dict(), dir + "/model.pth" )
            
        print("Validation AUC: {}". format(auc))
        # wandb.log({
        # "Train loss":losses.avg,
        # "AUC":auc,
        # "H-score":d_score / cnt
        # })
        f = open(dir + "/validation_performance.txt", "a")
        f.write(str(auc) + "\n")
        f.close()   

    return top1.avg

def test(test_loader, model, criterion, unorm, epoch, PATH, dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    global probs 
    global gt
    global k

    if args.dataset == 'CheXpert':
        class_num = 10
    elif args.dataset == 'NIH':
        class_num = 14

    test_loader_examples_num = len(test_loader.dataset)
    probs = np.zeros((test_loader_examples_num, class_num), dtype = np.float32)
    gt = np.zeros((test_loader_examples_num, class_num), dtype = np.float32)
    k = 0
    cnt = 0
    d_score = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        cnt += 1
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            # ResNet
            output, l1, l2, l3, hmap_t, hmaps_f  = model(inputs, target)
            # loss = criterion(output, target)+l1 + l2 + l3
            # DenseNet
            # output = model(inputs, parallel_last = False)
            loss = criterion(output, target)
        else:
            output, l1, l2, l3, hmap_t, hmaps_f, bw, h = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw
            d_score += h

        true_overlays = []
        false_overlays = []

        # for j in range(len(hmap_t)):
        #     true_overlays.append(wandb.Image(save_cam(inputs[j], hmap_t[j], i*len(inputs) + j ), caption="true"))
        #     false_overlays.append(wandb.Image(save_cam(inputs[j], hmaps_f[j], i*len(inputs) + j, conf=True), caption="false"))
        

        # wandb.log({
        #     'AttentionMap/True': true_overlays,
        #     'AttentionMap/False': false_overlays,
        # }) 

        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print("H-score: ", d_score / cnt)

    if args.dataset == 'ImageNet':
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        wandb.log({
            "Valid loss":losses.avg,
            "Valid Top 1 ACC":top1.avg,
            "Valid Top 5 ACC":top5.avg,
        })   
        f = open(dir + "/performance.txt", "a")
        f.write(str(top1.avg.item()) + "\n")
        f.close()
    elif args.dataset == 'CheXpert' or args.dataset == 'NIH': 
        auc = roc_auc_score(gt, probs)
        print("Test AUC: {}". format(auc))
        # wandb.log({
        # "Train loss":losses.avg,
        # "AUC":auc,
        # "H-score":d_score / cnt
        # })
        f = open(dir + "/test_performance.txt", "a")
        f.write(str(auc) + "\n")
        f.close()   

    return top1.avg



def save_checkpoint(state, is_best, path):
    filename='{}/checkpoint.pth.tar'.format(path)
    if is_best:
        torch.save(state, filename)


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

def save_cam(torch_img, grad_cam_map, index, conf = False):

    # print(grad_cam_map)
    grad_cam_map = grad_cam_map.unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data # (1, 1, W, H), min-max scaling

    #grad_cam_map = grad_cam_map.squeeze() # : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
    b, g, r = grad_heatmap.split(1)
    grad_heatmap = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.

    grad_result = grad_heatmap + torch_img.cpu() # (1, 3, W, H)
    grad_result = grad_result.div(grad_result.max()).squeeze() # (3, W, H)

    

    # if conf == False:
    #     save_image(grad_result,'C:/Users/hb/Desktop/code/ICASC++/result/CheXpert_plus(1)_training/result{}_true.png'.format(index))
    # else:
    #     save_image(grad_result,'C:/Users/hb/Desktop/code/ICASC++/result/CheXpert_plus(1)_training/result{}_false.png'.format(index))

    return grad_result


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
