import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
from torchsummary import summary



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        # an affine operation: y = Wx + b
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 5*5 from image dimension
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x,1)
        x = self.fc_layers(x)
        return x


class LeNet_3channel(nn.Module):
    def __init__(self):
        super(LeNet_3channel, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        # an affine operation: y = Wx + b
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 5*5 from image dimension
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x,1)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lenet1 = LeNet_style1()
    test_input = torch.rand(1, 3, 32, 32)
    test_output = lenet1(test_input)
    print(test_output.shape)
    print(test_output)
    
    lenet2 = LeNet_style2()
    test_input = torch.rand(1, 3, 32, 32)
    test_output = lenet2(test_input)
    print(test_output.shape)
    print(test_output)