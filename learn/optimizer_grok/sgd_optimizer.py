# ======================================
# 优化器：SGD with Momentum
# θ = θ - η * ∇L(θ) 其中 θ 是模型参数，η 是学习率，∇L(θ) 是损失函数的梯度
# Momentum 是一个动量项，它使得梯度下降过程更加平滑，即使在陡峭的山谷也能快速下降。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 模型定义
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器：SGD with Momentum
# lr=0.01: 适合 SGD 的初始学习率
# momentum=0.9: 加速梯度下降
# weight_decay=1e-4: L2 正则化，防止过拟合
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 学习率调度器：StepLR
# step_size=10: 每 10 epoch 调整一次
# gamma=0.1: 学习率乘以 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
num_epochs = 5  # 简化为 5 epoch 以便演示
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        running_loss += loss.item()
    
    # 计算平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 注意：scheduler.step() 在 optimizer.step() 后，epoch 结束时调用
    scheduler.step()
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

print("Training completed!")