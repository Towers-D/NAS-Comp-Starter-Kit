import torch
import torchvision.transforms as transforms
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train=False, transform=None):
        self.x = torch.tensor(x)

        # the test dataset has no labels, so we don't need to care about self.y
        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y)

        # example transform
        if train:
            self.mean = torch.mean(self.x, [0, 2, 3])
            self.std = torch.std(self.x, [0, 2, 3])
            self.transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        im = self.x[idx]

        if self.transform is not None:
            im = self.transform(im)

        # only return image in the case of the test dataloader
        if self.y is None:
            return im
        else:
            return im, self.y[idx]

class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata

    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!
        
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  
        
    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """
    def process(self):
        # create train, valid, and test datasets
        train_ds = Dataset(self.train_x, self.train_y, train=True)
        valid_ds = Dataset(self.valid_x, self.valid_y, transform=train_ds.transform)
        test_ds = Dataset(self.test_x, None, transform=train_ds.transform)

        batch_size = 64

        # build data loaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=batch_size,
                                                   drop_last=True,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)
        return train_loader, valid_loader, test_loader
