import torch


# FGSM攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


def pgd_attack(image, epsilon, alpha, data_grad):
    sign_data_grad = torch.sign(data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    if epsilon == 0:
        return perturbed_image
    delta = torch.clamp(perturbed_image - image, min=-alpha, max=alpha)
    perturbed_image = torch.clamp(image + delta, 0, 1)
    return perturbed_image
