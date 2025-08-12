import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

def get_cv_dataset(sargs: dict):
    """返回一个标准的 PyTorch Dataset 对象。"""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return datasets.ImageFolder(sargs["train_dir"], transform=transform)

class CVModel(nn.Module):
    def __init__(self, args, sargs):
        super().__init__()
        self.args = args
        self.sargs = sargs
        # 仅定义模型结构
        self.model = getattr(models, self.sargs["model_name"])(num_classes=self.args.num_classes)

    def get_optimizer_and_scheduler(self, model_to_optimize, sargs):
        """定义并返回优化器和学习率调度器。"""
        optimizer = optim.SGD(
            model_to_optimize.parameters(), 
            lr=sargs["base_lr"],
            momentum=sargs["momentum"], 
            weight_decay=sargs["wd"]
        )
        # CV 模型这里不使用 scheduler
        return optimizer, None

    def forward_pass(self, model, inputs, labels):
        """执行前向传播并返回损失。"""
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        return loss
    
    def forward(self, x):
        """nn.Module 需要的标准 forward 方法。"""
        return self.model(x)

    def print_info(self):
        print(f"Model: {self.sargs['model_name']}, Batch Size: {self.sargs['batch_size']}")