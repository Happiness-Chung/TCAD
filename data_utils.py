from torchvision import transforms, datasets
from datasets import ChexpertDataset,ChexpertValidationDataset, ChexpertTestDataset, NIHTrainDataset, NIHTestDataset
import torchvision
import torch
import random

def get_mean_var_classes(name):
    name = name.split('_')[-1]
    if name == 'cifar10':
       return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == 'cifar100':
       return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100
    elif name == 'stl10':
       return (0.4467, 0.43980, 0.4066), (0.2603, 0.2565, 0.2712), 10
    if name == 'CheXpert':
        return 0.485, 0.229, 10
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

def get_datasets(name, test = None):
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
                                    # transforms.RandomRotation(50),
                                    # transforms.RandomHorizontalFlip()
                                    ])
        # train = ChexpertTrainDataset(transform = transform, indices=list(range(sampling_num)))
        # validation = ChexpertValidationDataset(transform = test_transform)
        # train, validation = torch.utils.data.random_split(train, [int(0.9 * len(train)), len(train) - int(0.9 * len(train))])
        indices = list(range(224316))
        random.shuffle(indices)
        all = ChexpertDataset(transform = transform, indices=indices[:86336+25596])
        train, test = torch.utils.data.random_split(all, [len(all) - 25596, 25596])
        # test = ChexpertTestDataset(transform = transform)
        #test = ChexpertValidationDataset(transform = test_transform)
    elif name == 'NIH':
        sampling_num = 86336
        normalize = transforms.Normalize(mean, var)
        transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
        # if test == True:
        #     transform = transforms.Compose([
        #                             transforms.Resize([150,150]),
        #                             transforms.ToTensor()])
                                    
        train = NIHTrainDataset(data_dir='C:/Users/hb/Desktop/data/NIH', transform= transform, indices=list(range(sampling_num)))
        test = NIHTestDataset(data_dir='C:/Users/hb/Desktop/data/NIH', transform= transform)

        unorm = UnNormalize(mean, var)
        return train, test, test, num_classes, unorm
    
    unorm = UnNormalize(mean, var)

    return train, test, test, num_classes, unorm

