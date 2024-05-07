from torchvision import transforms, datasets
from datasets import ChexpertTrainDataset,ChexpertValidationDataset, ChexpertTestDataset, NIHTrainDataset, NIHTestDataset
import torchvision
import torch
import numpy as np


def get_mean_var_classes(name):
    name = name.split('_')[-1]
    if name == 'cifar10':
       return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == 'cifar100':
       return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100
    elif name == 'stl10':
       return (0.4467, 0.43980, 0.4066), (0.2603, 0.2565, 0.2712), 10
    if name == 'CheXpert':
        return 0.485, 0.229, 5
    if name == 'NIH':
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 14
    return None


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_datasets(args, test = None):
    name = args.dataset
    seed = args.seed
    mean, var, num_classes = get_mean_var_classes(name)
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, var)
            ])
        transform_test = transforms.Compose([transforms.ToTensor()
                                             , transforms.Normalize(mean, var)
                                             ])
        train = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, var)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, var)])
        train = datasets.CIFAR100(root='./data/', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR100(root='./data/', train=False, download=False, transform=transform_test)
    elif name == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, var)])
        transform_test = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, var)])
        train = datasets.STL10(root='./data/', split='train', download=False, transform=transform_train)
        test = datasets.STL10(root='./data/', split='test', download=False, transform=transform_test)
    elif name == 'CheXpert':
        sampling_num = 86336
        normalize = transforms.Normalize(mean=[mean],
                                 std=[var])
        transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize,
                                    #transforms.RandomRotation(50),
                                    # transforms.RandomHorizontalFlip()
                                    ])
        test_transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
        if test == True:
            transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor()
                                    ])
        
        split = f'./splits/{name}/{name}_split{seed}.txt'
        subject_order = open(split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train_subjects' in line for line in subject_order])
        valid_index = np.argmax(['valid_subjects' in line for line in subject_order])
        test_index = np.argmax(['test_subjects' in line for line in subject_order])
        train_names = subject_order[train_index + 1:valid_index] # Path
        valid_names = subject_order[valid_index + 1:test_index]
        test_names = subject_order[test_index + 1:]
        
        train = ChexpertTrainDataset(transform = transform, train_list = train_names)
        validation = ChexpertValidationDataset(transform = transform, valid_list = valid_names)
        test = ChexpertTestDataset(transform = test_transform, test_list = test_names)        
        
    elif name == 'NIH':
        sampling_num = 86336
        normalize = transforms.Normalize(mean, var)
        transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    # normalize
                                    ])
        # if test == True:
        #     transform = transforms.Compose([
        #                             transforms.Resize([150,150]),
        #                             transforms.ToTensor()])
                                    
        train = NIHTrainDataset(data_dir='C:/Users/hb/Desktop/data/NIH', transform= transform, indices=list(range(sampling_num)))
        test = NIHTestDataset(data_dir='C:/Users/hb/Desktop/data/NIH', transform= transform)
    
    unorm = UnNormalize(mean, var)

    return train, validation, test, num_classes, unorm

