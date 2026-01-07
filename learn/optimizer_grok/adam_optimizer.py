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

# 优化器：Adam
# lr=0.001: Adam 的标准初始学习率
# betas=(0.9, 0.999): 一阶和二阶动量参数
# weight_decay=1e-4: L2 正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# 学习率调度器：CosineAnnealingLR
# T_max=5: 在 5 epoch 内完成一次余弦退火周期
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 注意：scheduler.step() 在 epoch 结束时调用
    scheduler.step()
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

print("Training completed!")