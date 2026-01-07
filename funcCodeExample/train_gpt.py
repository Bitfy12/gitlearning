import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # ✅ 进度条显示库

# =========================================
# 1️⃣ 基本超参数与设备配置
# =========================================
batch_size = 128              # 每个 batch 中的样本数量
learning_rate = 0.01          # 初始学习率
num_epochs = 10               # 训练的总轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU

# =========================================
# 2️⃣ 数据加载与预处理
# =========================================
# transforms.Compose：顺序执行多个预处理操作
# transforms.Normalize(mean, std)：将像素值标准化到 [-1,1] 区间，提高训练稳定性
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图片转换为Tensor，并自动缩放到 [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10：10类彩色图像数据集 (32x32)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader：将数据集分批打包、打乱，并支持多线程加载
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
# ⚠️ 注意：
# - 训练集一定要 shuffle=True，防止模型记忆数据顺序。
# - num_workers 根据 CPU 核心数设置，否则可能太慢或卡死。

# =========================================
# 3️⃣ 定义模型结构（简单 CNN）
# =========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 两层卷积 + 池化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 输入通道=3, 输出通道=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32→64
        self.pool = nn.MaxPool2d(2, 2)                            # 下采样，尺寸减半
        self.fc1 = nn.Linear(64 * 8 * 8, 128)                     # 全连接层
        self.fc2 = nn.Linear(128, 10)                             # 输出层（10 类）
        self.relu = nn.ReLU()                                     # 激活函数 ReLU

    def forward(self, x):
        # 前向传播：定义数据流动的计算路径
        x = self.pool(self.relu(self.conv1(x)))  # [B,3,32,32] → [B,32,16,16]
        x = self.pool(self.relu(self.conv2(x)))  # [B,32,16,16] → [B,64,8,8]
        x = x.view(x.size(0), -1)                # 展平：变成 [B, 64*8*8]
        x = self.relu(self.fc1(x))               # 隐藏层
        x = self.fc2(x)                          # 输出层 (未经过 softmax)
        return x

# 将模型加载到 GPU（或 CPU）
model = SimpleCNN().to(device)

# =========================================
# 4️⃣ 定义损失函数、优化器与学习率调度器
# =========================================
criterion = nn.CrossEntropyLoss()  # 分类任务的常用损失函数

# SGD 优化器：带 momentum（动量）和 weight_decay（L2 正则）
optimizer = optim.SGD(
    model.parameters(), 
    lr=learning_rate, 
    momentum=0.9, 
    weight_decay=5e-4
)

# StepLR：每过 step_size 个 epoch 学习率乘以 gamma
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# =========================================
# 5️⃣ 训练与验证循环（加入 tqdm 进度条）
# =========================================
for epoch in range(num_epochs):
    # -------- 训练阶段 --------
    model.train()  # 启用训练模式（启用 dropout / BN 的更新）
    running_loss = 0.0

    # tqdm：包装训练数据加载器，显示实时进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)

    for inputs, targets in progress_bar:
        # 将数据搬到 GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # ---- 反向传播与参数更新 ----
        optimizer.zero_grad()   # ⚠️ 清空上一次累积的梯度（必须！）
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 更新参数（执行一次梯度下降）

        # 累积损失（用于计算 epoch 平均值）
        running_loss += loss.item()

        # 在 tqdm 进度条中实时显示当前 batch 的损失与学习率
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    # ⚠️ 注意：scheduler.step() 一定要放在 optimizer.step() 之后！
    scheduler.step()

    # -------- 验证阶段 --------
    model.eval()  # 推理模式（冻结 dropout 与 BN 的均值方差）
    correct = 0
    total = 0
    val_loss = 0.0

    # tqdm：包装验证集，显示进度条
    val_bar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)

    with torch.no_grad():  # 禁用梯度计算，节省显存 + 加快推理
        for inputs, targets in val_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 计算分类准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新 tqdm 进度条信息
            val_bar.set_postfix(val_loss=loss.item())

    acc = 100. * correct / total  # 验证集准确率

    # 每个 epoch 打印一次总结信息
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {running_loss/len(train_loader):.4f} "
          f"| Val Loss: {val_loss/len(test_loader):.4f} "
          f"| Val Acc: {acc:.2f}% "
          f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

print("🎯 Training complete.")
