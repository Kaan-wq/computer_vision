import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2), nn.BatchNorm2d(64), nn.ReLU())

        # First Residual Block
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_x = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))

        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_x_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))

        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_x_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))

        # Second Residual Block
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3_x = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128))

        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3_x1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128))

        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3_x_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128))

        self.conv3_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3_x_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128))

        # Third Residual Block
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        self.conv4_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        self.conv4_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        self.conv4_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        self.conv4_4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x_4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        self.conv4_5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv4_x_5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        # Fourth Residual Block
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(512), nn.ReLU())
        self.conv5_x = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512))

        self.conv5_1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv5_x_1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512))

        self.conv5_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv5_x_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512))


        self.skip3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
        self.skip4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2), nn.BatchNorm2d(256))
        self.skip5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2), nn.BatchNorm2d(512))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 120), nn.ReLU(), nn.Dropout(0.5) ,nn.Linear(120, 10))

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.relu(self.conv2_x(self.conv2(x) + x))
        x = self.relu(self.conv2_x_1(self.conv2_1(x) + x))
        x = self.relu(self.conv2_x_2(self.conv2_2(x) + x))

        x = self.relu(self.conv3_x(self.conv3(x) + self.skip3(x.clone())))
        x = self.relu(self.conv3_x1(self.conv3_1(x) + x))
        x = self.relu(self.conv3_x_2(self.conv3_2(x) + x))
        x = self.relu(self.conv3_x_3(self.conv3_3(x) + x))

        x = self.relu(self.conv4_x(self.conv4(x) + self.skip4(x.clone())))
        x = self.relu(self.conv4_x_1(self.conv4_1(x) + x))
        x = self.relu(self.conv4_x_2(self.conv4_2(x) + x))
        x = self.relu(self.conv4_x_3(self.conv4_3(x) + x))
        x = self.relu(self.conv4_x_4(self.conv4_4(x) + x))
        x = self.relu(self.conv4_x_5(self.conv4_5(x) + x))

        x = self.relu(self.conv5_x(self.conv5(x) + self.skip5(x.clone())))
        x = self.relu(self.conv5_x_1(self.conv5_1(x) + x))
        x = self.relu(self.conv5_x_2(self.conv5_2(x) + x))

        x = self.global_avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return x
    
    def train_model(self, trainloader, testloader, criterion, optimizer, epochs=2):
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
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