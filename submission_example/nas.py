import torchvision
import torch.nn as nn

class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """
    def __init__(self, train_loader, valid_loader, metadata):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata


    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):
        num_classes = self.metadata['num_classes']

        # load resnet18 model
        model = torchvision.models.resnet18()

        # reshape it to the data
        model.conv1 = nn.Conv2d(self.metadata['input_shape'][1], 64, kernel_size=(3, 3), stride=1, padding=1)
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        return model