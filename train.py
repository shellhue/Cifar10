import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet56


# super parameters
EPOCH = 164
BATCH_SIZE = 128
LR = 0.1


# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training device is: ")

# data preparing
# data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # random crop with 4 padding
    transforms.RandomHorizontalFlip(),  # random horizontal flip
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # normalize rgb
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

train_size = int(1 * len(full_dataset))
test_size = len(full_dataset) - train_size
training_set, validation_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # labels


# create resnet model
resnet20 = ResNet56().to(device)

criterion = nn.CrossEntropyLoss()  # use cross entropy loss
# sgd optimizer with momentum and L2 regulation
optimizer = optim.SGD(resnet20.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)


# train
if __name__ == "__main__":
    print("Start Training!")
    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        resnet20.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(training_loader, 0):
            length = len(training_loader)
            iteration = i + 1 + epoch * length
            if iteration == 32000 or iteration == 48000:
                LR = LR / 10
                optimizer = optim.SGD(resnet20.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward -> backward -> update gradients
            outputs = resnet20(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | LearingRate: %.6f | Acc: %.3f%% '
                  % (epoch + 1, iteration, sum_loss / (i + 1), LR, 100. * float(correct) / float(total)))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            # calculate validation accuraccy, no needs to do so, as there is no super parameter to choose
            if False:
                correct = 0
                total = 0
                for data in val_loader:
                    resnet20.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = resnet20(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('val acc：%.3f%%' % (100 * float(correct) / float(total)))

            # calculate test error
            if False:
                correct = 0
                total = 0
                for data in test_loader:
                    resnet20.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = resnet20(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('test acc：%.3f%%' % (100 * float(correct) / float(total)))
                print('Saving model......')
                torch.save(resnet20.state_dict(), '%s/resnet56_%03d.pth' % ("./log", epoch + 1))
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
