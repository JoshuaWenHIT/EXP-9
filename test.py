import torch
import torch.nn.functional as F
from attack import fgsm_attack, pgd_attack


def test(model, device, test_loader, epsilon, attack='pgd', step=1):
    # 精度计数器
    corrent = 0  # 正确的数量
    adv_examples = []  # 存储攻击成功的样本
    perturbed_data = []
    # 循环遍历测试集中的所有示例
    for data, target in test_loader:
        # 将数据和标签发送到设备
        data, target = data.to(device), target.to(device)
        if attack == 'fgsm':
            # 设置张量的requires_grad属性，这对于攻击很关键
            data.requires_grad = True
            # 通过模型前向传递数据
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # 获取初始预测结果
            # 如果初始预测是错误的，不打断攻击，继续
            if init_pred.item() != target.item():
                continue
            # 计算损失
            loss = F.nll_loss(output, target)
            # 将所有现有的渐变归零，作用是清除上一次的梯度
            model.zero_grad()
            # 计算后向传递模型的梯度,计算出各个参数的梯度
            loss.backward()
            # 收集datagrad 为了攻击
            data_grad = data.grad.data
            # 调用FGSM攻击
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack == 'pgd':
            data.requires_grad = True
            # output = model(data)
            # data = data + 0.001 * torch.randn(data.shape)
            # data = torch.clamp(data, 0 - epsilon, 1 + epsilon)
            for _ in range(step):
                # 设置张量的requires_grad属性，这对于攻击很关键
                # data.requires_grad = True
                # 通过模型前向传递数据
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1]  # 获取初始预测结果
                # 如果初始预测是错误的，不打断攻击，继续
                if init_pred.item() != target.item():
                    continue
                # 计算损失
                loss = F.nll_loss(output, target)
                # 将所有现有的渐变归零，作用是清除上一次的梯度
                data_grad = torch.autograd.grad(loss, data, create_graph=True)[0]
                # model.zero_grad()
                # 计算后向传递模型的梯度,计算出各个参数的梯度
                # loss.backward()
                # 收集datagrad 为了攻击
                # data_grad = data.grad.data
                data = pgd_attack(data, epsilon=epsilon, alpha=1, data_grad=data_grad)
            perturbed_data = data
            # perturbed_data = pgd_attack(data, epsilon, data_grad)

        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1]  # 获取最终预测结果
        if final_pred.item() == target.item():
            corrent += 1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:  # 保存epsilon>0的样本
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # 计算最终的正确率
    final_acc = corrent / float(len(test_loader))
    print("Epsilon:{}\tTest Accuracy={}/{}={}".format(epsilon, corrent, len(test_loader), final_acc))

    # 返回正确率和对抗样本
    return final_acc, adv_examples
