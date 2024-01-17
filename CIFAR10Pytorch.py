import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from ResNet18Pytorch import ResNet18
from torchsummary import summary

# load data
train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
print(train_dataloader.dataset)
print(test_dataloader.dataset)

# model construction
model = ResNet18(10, pre_filter_size=5)
model.cuda()
summary(model, (3,32,32))
loss_function = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.001)

for i in range(10):
    print("-------epoch  {} -------".format(i + 1))
    # 训练步骤
    model.train()
    for step, [imgs, targets] in enumerate(train_dataloader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_function(outputs, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_step = len(train_dataloader) * i + step + 1
        if train_step % 100 == 0:
            print("train time：{}, Loss: {}".format(train_step, loss.item()))

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("test set Loss: {}".format(total_test_loss))
    print("test set accuracy: {}".format(total_accuracy / len(test_data)))
    # torch.save(model, "models/module_{}.pth".format(i + 1))
    # print("saved epoch {}".format(i + 1))

