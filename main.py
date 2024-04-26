import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from test import test
from network import Net, use_cuda, pretrained_model

epsilons = [0, .05, .1, .15, .2, .25, .3]  # epsilon值
# epsilons = []
accuracies = []
examples = []

# MINIST数据集测试和加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)
# 查看是否配置GPU，没有就调用CPU
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 对于每个epsilon，运行测试
for eps in epsilons:
    # acc, ex = test(model, device, test_loader, eps)
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")  # 准确率与epsilon的关系
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# 画出几个epsilon的示例
cnt = 0  # 计数器
plt.figure(figsize=(8, 10))  # 画布大小
for i in range(len(epsilons)):  # 遍历epsilon
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:  # 第一行的标题
            plt.ylabel("Eps:{}".format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]  # 获取原始，对抗，样本
        plt.title("{} -> {}".format(orig, adv), color=("green" if orig == adv else "red"), fontsize=14)
        plt.imshow(ex, cmap="gray")
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
