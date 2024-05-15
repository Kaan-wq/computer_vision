import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()

        if in_channels == out_channels:
            stride = 1

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1: 
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.skip(identity)
        out = self.relu(out)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(ResBlock, self).__init__()

        assert num_blocks > 0, 'Number of blocks should be greater than 0'
        assert in_channels > 0, 'Input channels should be greater than 0'
        assert out_channels > 0, 'Output channels should be greater than 0'

        blocks = []
        for i in range(num_blocks):
            blocks.append(BasicBlock(in_channels if i == 0 else out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
    

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2), nn.BatchNorm2d(64), nn.ReLU())

        # First Residual Block
        self.conv2 = ResBlock(64, 64, num_blocks=3)

        # Second Residual Block
        self.conv3 = ResBlock(64, 128, num_blocks=4)

        # Third Residual Block
        self.conv4 = ResBlock(128, 256, num_blocks=6)

        # Fourth Residual Block
        self.conv5 = ResBlock(256, 512, num_blocks=3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 120), nn.ReLU(), nn.Dropout(0.5) ,nn.Linear(120, 10))

    def forward(self, x):
        
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x
    
    def train_model(self, trainloader, testloader, criterion, optimizer, epochs=2):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1} Progress')):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.2f}')
                    running_loss = 0.0
                
            print('\n')
            print(f'Eval loss : {self.eval_model(testloader, criterion):.2f}')
            print('\n')

            torch.save(self.state_dict(), 'models/cifar_resnet34.pth')

        print('Finished Training')

    def eval_model(self, testloader, criterion):
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.forward(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
        return running_loss / len(testloader)
    

class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2), nn.BatchNorm2d(64), nn.ReLU())

        # First Residual Block
        self.conv2 = ResBlock(64, 64, num_blocks=2)

        # Second Residual Block
        self.conv3 = ResBlock(64, 128, num_blocks=2)

        # Third Residual Block
        self.conv4 = ResBlock(128, 256, num_blocks=2)

        # Fourth Residual Block
        self.conv5 = ResBlock(256, 512, num_blocks=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 120), nn.ReLU(), nn.Dropout(0.5) ,nn.Linear(120, 10))

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return x
    
    def train_model(self, trainloader, testloader, criterion, optimizer, epochs=2):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1} Progress')):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.2f}')
                    running_loss = 0.0
                
            print('\n')
            print(f'Eval loss : {self.eval_model(testloader, criterion):.2f}')
            print('\n')

            torch.save(self.state_dict(), 'models/cifar_resnet8.pth')

        print('Finished Training')

    def eval_model(self, testloader, criterion):
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.forward(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
        return running_loss / len(testloader)