import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 20, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second block
        self.conv3 = nn.Conv2d(20, 28, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(28)
        self.conv4 = nn.Conv2d(28, 28, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(28)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Added new convolutional layer before final classification
        self.conv_extra = nn.Conv2d(28, 14, 3, padding=1)
        self.bn_extra = nn.BatchNorm2d(14)
        
        # Final classification block
        self.conv5 = nn.Conv2d(14, 10, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.15)

        print(f'Total parameters: {sum(p.numel() for p in self.parameters())}')

    def forward(self, x):
        # First block with residual connection
        x1 = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x1)))
        x = x + x1
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual connection
        x2 = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x2)))
        x = x + x2
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Added extra convolution with ReLU
        x = F.relu(self.bn_extra(self.conv_extra(x)))
        
        # Focused classification
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)