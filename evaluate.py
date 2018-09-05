import argparse
import resnet
import torchvision
import torchvision.transforms as transforms
import torch

# argument parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Model Converting to proto buffer')
parser.add_argument('--weights', help='weights file path')
parser.add_argument('--layers', default=20, help="number of layers of the trained model")

args = parser.parse_args()


WEIGHTS = args.weights
LAYERS = int(args.layers)

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if LAYERS == 56:
    model = resnet.ResNet56().to(device)
else:
    model = resnet.ResNet20().to(device)


print("start loading pretrained weights")
model.load_state_dict(torch.load(WEIGHTS))
model.train(False)
print("finish loading pretrained weights")


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)


def _main():
    print("Waiting Test!")
    with torch.no_grad():
        # calculate test error
        correct = 0
        total = 0
        for data in test_loader:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('test acc：%.3f%%' % (100 * float(correct) / float(total)))


# train
if __name__ == "__main__":
    _main()