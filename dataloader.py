import os
import PIL
import math
import torch
import pickle
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data.dataset import Subset




class MVTecADDataset(Dataset):
    def __init__(self, root_dir, categories, transform, phase='train'):
        self.root_dir = root_dir
        self.categories = categories
        self.phase = phase
        
        self.image_paths = []
        self.targets = []

        self.transform = transform

        self._load_dataset()

    def _load_dataset(self):
        # Set the paths for training and test datasets
        phase_dir = 'train' if self.phase == 'train' else 'test'
        # category_path = os.path.join(self.root_dir, self.categories, phase_dir)

        for l, category in enumerate(self.categories):
            category_path = os.path.join(self.root_dir + 'mvtec_ad/', category, phase_dir)

            for class_name in os.listdir(category_path):
                class_dir = os.path.join(category_path, class_name)
                if not os.path.isdir(class_dir):
                    continue

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    # Label: 0 for 'good' images, 1 for 'anomaly' images
                    self.targets.append(0 if class_name == 'good' else 1)
                    # self.targets.append(l)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class batch_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(f'{self.path}/batch_{idx}.pkl', 'rb') as f:
            return torch.tensor(pickle.load(f)).float()
        

class SVHN(Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"],
        'train_and_extra': [
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]
            self.filename = self.split_list[split][0][1]
            self.file_md5 = self.split_list[split][0][2]
        else:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        if self.split == "test":
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']

            if self.split == "train_and_extra":
                extra_filename = self.split_list[split][1][1]
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))
                self.data = np.concatenate([self.data,
                                                  loaded_mat['X']], axis=3)
                self.targets = np.vstack((self.targets,
                                               loaded_mat['y']))
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == "test":
            return len(self.data)
        else:
            return len(self.data)

    def _check_integrity(self):
        root = self.root
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            fpath = os.path.join(root, self.filename)
            train_integrity = check_integrity(fpath, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            fpath = os.path.join(root, extra_filename)
            return check_integrity(fpath, md5) and train_integrity
        else:
            md5 = self.split_list[self.split][2]
            fpath = os.path.join(root, self.filename)
            return check_integrity(fpath, md5)

    def download(self):
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            download_url(self.url, self.root, self.filename, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            download_url(self.url, self.root, extra_filename, md5)
        else:
            md5 = self.split_list[self.split][2]
            download_url(self.url, self.root, self.filename, md5)


class load_np_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform, dataset, train=True, anomaly_path=None):
        self.dataset = dataset
        if train and dataset == 'cifar10':
            self.data = np.load(imgs_path)[:50000]
            self.targets = np.load(targets_path)[:50000]
        elif dataset == 'cifar10':
            self.data = np.load(imgs_path)[:10000]
            self.targets = np.load(targets_path)[:10000]
        elif dataset == 'anomaly':
            self.data = np.load(imgs_path)[:10000]
            self.targets = np.load(targets_path)[:10000]
            self.anomaly = np.load(anomaly_path)
        else:
            self.data = np.load(imgs_path)
            self.targets = np.load(targets_path)

        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        img = self.transform(img)
        if self.dataset == 'anomaly':
            return img, target, self.anomaly[idx]
        else:
            return img, target


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def sparse2coarse(targets):
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def noise_loader(args, batch_size=64, num_workers=0, one_class_idx=None, coarse=True, dataset='cifar10', preprocessing='clip', k_pairs=1, resize=224):
    # Filling paths
    np_train_target_path = os.path.join(args.config['generalization_path'], f'{dataset}_Train_s1/labels.npy')
    np_test_target_path = os.path.join(args.config['generalization_path'], f'{dataset}_Test_s5/labels.npy')
    np_train_root_path = os.path.join(args.config['generalization_path'], f'{dataset}_Train_s1')
    np_test_root_path = os.path.join(args.config['generalization_path'], f'{dataset}_Test_s5')
        
    if args.dataset == 'mvtec_ad' and resize==32:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(math.ceil(resize*1.14)),
            torchvision.transforms.CenterCrop(resize),
            torchvision.transforms.ToTensor()])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(math.ceil(resize*1.14)),
            torchvision.transforms.CenterCrop(resize),
            torchvision.transforms.ToTensor()])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

    with open(f'./ranks/{preprocessing}/{dataset}/wasser_dist_softmaxed.pkl', 'rb') as file:
        probs = pickle.load(file)

    print("Creating noises loader")

    train_positives_loader = []
    test_positives_loader = []
    train_negetives_loader = []
    test_negetives_loader = []
    all_train_positives_datasets = []
    all_train_negetives_datasets = []
    all_test_positives_datasets = []
    all_test_negetives_datasets = []
    all_train_dataset_positives_one_class = []
    all_train_dataset_negetives_one_class = []
    all_test_dataset_positives_one_class = []
    all_test_dataset_negetives_one_class = []
    
    # Loading positive
    for k in range(1, k_pairs + 1):
        noise = list(probs[one_class_idx].keys())[k].replace('dist_','')
        print(f"Selecting {noise} as positive pair for class {one_class_idx}")
        np_train_img_path = os.path.join(np_train_root_path, noise + '.npy')
        train_positives_datasets = load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True)
        if dataset == 'cifar100' and coarse:
            train_positives_datasets.targets = sparse2coarse(train_positives_datasets.targets)

        np_test_img_path = os.path.join(np_test_root_path, noise + '.npy')
        test_positives_datasets = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset, train=False)
        if dataset == 'cifar100' and coarse:
            test_positives_datasets.targets = sparse2coarse(test_positives_datasets.targets)

        all_train_positives_datasets.append(train_positives_datasets)
        all_test_positives_datasets.append(test_positives_datasets)
        
        if one_class_idx != None:
            all_train_dataset_positives_one_class.append(get_subclass_dataset(train_positives_datasets, one_class_idx))
            all_test_dataset_positives_one_class.append(get_subclass_dataset(test_positives_datasets, one_class_idx))
        else:
            all_train_dataset_positives_one_class.append(train_positives_datasets)
            all_test_dataset_positives_one_class.append(test_positives_datasets)

        train_positives_loader.append(DataLoader(all_train_dataset_positives_one_class[k-1], shuffle=False, batch_size=batch_size, num_workers=num_workers))
        test_positives_loader.append(DataLoader(all_test_dataset_positives_one_class[k-1], shuffle=False, batch_size=batch_size, num_workers=num_workers))

    # Loading negative 
    for k in range(1, k_pairs + 1):
        noise = list(probs[one_class_idx].keys())[-k].replace('dist_', '')
        print(f"Selecting {noise} as negetive pair for class {one_class_idx}")
        np_train_img_path = os.path.join(np_train_root_path, noise + '.npy')
        train_negetives_datasets = load_np_dataset(np_train_img_path, np_train_target_path, train_transform, dataset, train=True)
        if dataset == 'cifar100':
            train_negetives_datasets.targets = sparse2coarse(train_negetives_datasets.targets)
        
        np_test_img_path = os.path.join(np_test_root_path, noise + '.npy')
        test_negetives_datasets = load_np_dataset(np_test_img_path, np_test_target_path, test_transform, dataset, train=False)
        if dataset == 'cifar100':
            test_negetives_datasets.targets = sparse2coarse(test_negetives_datasets.targets)

        all_train_negetives_datasets.append(train_negetives_datasets)
        all_test_negetives_datasets.append(test_negetives_datasets)
        
        if one_class_idx != None:
            all_train_dataset_negetives_one_class.append(get_subclass_dataset(train_negetives_datasets, one_class_idx))
            all_test_dataset_negetives_one_class.append(get_subclass_dataset(test_negetives_datasets, one_class_idx))
        else:
            all_train_dataset_negetives_one_class.append(train_negetives_datasets)
            all_test_dataset_negetives_one_class.append(test_negetives_datasets)
        
        train_negetives_loader.append(DataLoader(all_train_dataset_negetives_one_class[k-1], shuffle=False, batch_size=batch_size, num_workers=num_workers))
        test_negetives_loader.append(DataLoader(all_test_dataset_negetives_one_class[k-1], shuffle=False, batch_size=batch_size, num_workers=num_workers))

    return train_positives_loader, train_negetives_loader, test_positives_loader, test_negetives_loader


def load_imagenet(path, batch_size=64, num_workers=0, one_class_idx=None):
    print('loading Imagenet')
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()])

    train_data = torchvision.datasets.ImageNet(root=os.path.join(path, 'ImageNet'), split='train', transform=transform)
    val_data = torchvision.datasets.ImageNet(root=os.path.join(path, 'ImageNet'), split='val', transform=transform)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        val_data = get_subclass_dataset(val_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def load_cifar10(path, batch_size=64, num_workers=0, one_class_idx=None):
    print('loading cifar10')
    data_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(
        path, train=True, transform=data_transforms, download=True)
    test_data = torchvision.datasets.CIFAR10(
        path, train=False, transform=data_transforms, download=True)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def load_svhn(path, batch_size=64, num_workers=0, one_class_idx=None):
    print('loading SVHN')
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = SVHN(root=path, split="train", transform=transform)
    test_data = SVHN(root=path, split="test", transform=transform)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def load_cifar100(path, batch_size=64, num_workers=0, one_class_idx=None, coarse=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=transform)
    
    if coarse:
        train_data.targets = sparse2coarse(train_data.targets)
        test_data.targets = sparse2coarse(test_data.targets)

    if one_class_idx != None:
        train_data = get_subclass_dataset(train_data, one_class_idx)
        test_data = get_subclass_dataset(test_data, one_class_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


def load_mvtec_ad(path, resize=224, batch_size=64, num_workers=0, one_class_idx=None, coarse=True):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(math.ceil(resize*1.14)),
            torchvision.transforms.CenterCrop(resize),
            torchvision.transforms.ToTensor()])
    
    cc = ['bottle', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    if one_class_idx != None:
        print(cc[one_class_idx])
        categories = [cc[one_class_idx]]
        
    else:
        categories = ['bottle', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    train_data = MVTecADDataset(path, transform=transform, categories=categories, phase='train')
    test_data = MVTecADDataset(path, transform=transform, categories=categories, phase='test')
    
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader

