import os
import argparse
import warnings
import json

from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import random
import itertools
from datetime import datetime
random.seed(datetime.now())

"""

[Directory tree]
data/
    night_test/
    day_test/
    night_train/
        night/
            1.jpg
            2.jpg
        ...
    day_train/
        day/
            1.jpg
            2.jpg
        ...
    night_queries/
data_loader.py
...

"""

IMG_TYPE_LIST = ["day","night"]

def get_mean_std_pic():
    """ Compute running mean and std.
        Read all images would crash memory.
    """
    print("[Computing mean and std of all data...]")
    print()
    
    stats = { "running_means":np.array([0.,0.,0.]), "running_std":np.array([0.,0.,0.]),
            "day_means":np.array([0.,0.,0.]), "night_means":np.array([0.,0.,0.]),
            "day_std":np.array([0.,0.,0.]), "night_std":np.array([0.,0.,0.]),
            "num_of_imgs":0, "num_of_dimgs":0, "num_of_nimgs":0
             }
    
    for img_type in IMG_TYPE_LIST:
        train_path = os.path.join('./data', '{}_train'.format(img_type))
        train_path += '/{}'.format(img_type)
        train_path = os.path.abspath(train_path)
        
        img_shape = cv2.imread(os.path.join(train_path, os.listdir(train_path)[0]), cv2.IMREAD_COLOR).shape
        
        # Load in the images
        for filename in tqdm(os.listdir(train_path), desc="loading {} images".format(img_type)):
            if (filename[-3:] != "jpg"):
                continue
                
            img_file = os.path.join(train_path, filename)
            img = cv2.imread(img_file,cv2.IMREAD_COLOR)
            # print(img[:,:,0].shape) # (512,512,3)
            # print(np.max(img)) # 255
            # print(np.min(img)) # 0
            
            # convert dtype from uint8 to uint16
            
            img = np.array(img,dtype=np.uint16)
            
            # running_means = sum(mean of channels each pic) / num_of_pic
            # running_std = (sum(squared sum/255**2) / (pixels_per_pic * num_of_pic) - mean_squared) ** (1/2)
            stats["running_means"] += np.mean(img,(0,1)) / 255
            stats["running_std"] += np.sum((img**2 / 255**2),(0,1)) / (img.shape[0]*img.shape[1])
            if (img_type=="day"):
                stats["day_means"] += np.mean(img,(0,1)) / 255
                stats["day_std"] += np.sum(img**2 / 255**2,(0,1)) / (img.shape[0]*img.shape[1])
                stats["num_of_dimgs"] += 1
            else:
                stats["night_means"] += np.mean(img,(0,1)) / 255
                stats["night_std"] += np.sum((img**2 / 255**2),(0,1)) / (img.shape[0]*img.shape[1])
                stats["num_of_nimgs"] += 1
            
            stats["num_of_imgs"] += 1
        
        if (img_type=="day"):
            stats["day_means"] /= stats["num_of_dimgs"]
            stats["day_std"] /= stats["num_of_dimgs"]
            stats["day_std"] -= stats["day_means"]**2
            stats["day_std"] = stats["day_std"]**(1/2)
        else:
            stats["night_means"] /= stats["num_of_nimgs"]
            stats["night_std"] /= stats["num_of_nimgs"]
            stats["night_std"] -= stats["night_means"]**2
            stats["night_std"] = stats["night_std"]**(1/2)
            
    stats["running_means"] /= stats["num_of_imgs"]
    stats["running_std"] /= stats["num_of_imgs"]
    stats["running_std"] -= stats["running_means"]**2
    stats["running_std"] = stats["running_std"]**(1/2)
    
    print()
    print("=========== Info ===========")
    print("Number of images: ", stats["num_of_imgs"])
    print()
    print("All Means: ", stats["running_means"])
    print("Only Nighttime Means: ", stats["night_means"])
    print("Only Daytime Means: ", stats["day_means"])
    print()
    print("All Std: ", stats["running_std"])
    print("Only Nighttime Std: ", stats["night_std"])
    print("Only Daytime Std: ", stats["day_std"])
    print("===========      ===========")
    print()
    
    return stats
    

def get_image_loader(img_type, opts, mean_pics=[0.5,0.5,0.5], std_pics=[0.5,0.5,0.5]):
    """ Creates training and test data loaders.
        mean_pics = pass # (ch1, ch2, ch3)
        std_pics = pass # (ch1, ch2, ch3)
    """
    
    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_pics, std_pics)
                ])
    
    train_path = os.path.join('./drive/MyDrive/workspace_4471/data', '{}_train'.format(img_type))
    test_path = os.path.join('./drive/MyDrive/workspace_4471/data', '{}_test'.format(img_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    # print(len(train_dataset)) # 11550
    test_dataset = datasets.ImageFolder(test_path, transform)
    
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader

def get_test_image_loader(img_type, opts, mean_pics=[0.5,0.5,0.5], std_pics=[0.5,0.5,0.5]):
    """ Creates training and test data loaders.
        mean_pics = pass # (ch1, ch2, ch3)
        std_pics = pass # (ch1, ch2, ch3)
    """
    
    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_pics, std_pics)
                ])

    # print(len(train_dataset)) # 11550
    test_dataset = datasets.ImageFolder('./drive/MyDrive/4471train/image_pair/night_images', transform)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return test_dloader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser

if __name__ == '__main__':
    
    img_stats = get_mean_std_pic()
    
    
    '''
    parser = create_parser()
    opts = parser.parse_args()
    
    train_loader, test_loader = get_image_loader("day", opts)
    print("Num of batches in train_loader: ", len(train_loader))
    for batch in train_loader:
        
        print("Num of imgs in batch: ", )
        print("batch[0] (-> image) dim: ", batch[0].shape) # torch.Size([16, 3, 512, 512])
        print("batch[1] (-> dont care) dim: ", batch[1].shape) # torch.Size([16])
        break
        '''
    
