import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 注意事项1: 设备选择 - 优先使用 GPU 以加速训练。如果没有 GPU，fallback 到 CPU。
# 确保在模型和数据上都使用 .to(device) 以避免 CPU/GPU 混合错误。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 步骤1: 数据集加载和预处理
# 使用 torchvision 下载 MNIST 数据集（28x28 灰度图像，10 类）。
# transforms: Normalize 归一化数据（均值 0.5，标准差 0.5），帮助模型收敛更快。
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像转换为 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化：(x - mean) / std
])

# 下载并加载训练集和测试集
# 注意事项2: root 参数指定下载路径；download=True 自动下载；train=True/False 区分训练/测试。
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 步骤2: 数据加载器 (DataLoader)
# batch_size: 批量大小，通常 32-256，根据内存调整。太大可能导致内存溢出，太小训练慢。
# shuffle: 训练集 shuffle=True 以随机化顺序，避免过拟合；测试集 shuffle=False 以保持顺序。
# num_workers: 多线程加载数据，加速 I/O（根据 CPU 核心设置，0 表示单线程）。
# 注意事项3: 如果使用 GPU，确保 num_workers > 0 以并行加载数据，避免 I/O 瓶颈。
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 步骤3: 模型定义
# 一个简单的 MLP：输入 28*28=784，隐藏层 128，输出 10 类。
# 注意事项4: 模型继承 nn.Module，必须实现 __init__ 和 forward。

# 使用 nn.ReLU() 激活函数，避免梯度消失。
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)     # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平图像：从 (batch, 1, 28, 28) 到 (batch, 784)
        x = torch.relu(self.fc1(x))  # 激活
        x = self.fc2(x)  # 输出（logits，不需 softmax，因为 CrossEntropyLoss 会处理）
        return x

model = SimpleNet().to(device)  # 将模型移到设备

# 步骤4: 损失函数
# CrossEntropyLoss: 用于多分类，内部包含 softmax 和 NLLLoss。
# 注意事项5: 对于分类任务，不要在 forward 中加 softmax，否则重复计算。
criterion = nn.CrossEntropyLoss()

# 步骤5: 优化器
# Adam: 自适应学习率，适合大多数任务。初始 lr=0.001。
# 注意事项6: optimizer 只优化 model.parameters()，确保模型已定义。
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 步骤6: 学习率调度器 (可选，但这里使用 StepLR)
# step_size=10: 每 10 epoch 调整一次；gamma=0.1: 乘以 0.1 (除以 10)。
# 注意事项7: 调度器必须在 optimizer 之后定义，因为它依赖 optimizer。
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 步骤7: 训练循环
# num_epochs: 总 epoch 数，根据任务调整（这里 20）。
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式（启用 dropout/batchnorm 等）
    
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 数据移到设备
        
        # 注意事项8: 每个 batch 开始前，必须 zero_grad() 清零梯度。
        # 否则梯度会累积，导致错误更新。
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 注意事项9: optimizer.step() 更新参数，必须在 backward() 之后。
        # 这是实际应用梯度的步骤。
        optimizer.step()
        
        running_loss += loss.item()
    
    # epoch 结束，计算平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 注意事项10: scheduler.step() 必须在 optimizer.step() 之后调用，通常在 epoch 结束时。
    # 因为它基于当前 epoch 更新 lr，并在下一个 epoch 使用新 lr。
    # 如果放在 batch 循环中，会导致 lr 过频更新（除非设计如此）。
    scheduler.step()
    
    # 可选: 当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6f}")
    
    # 可选: 验证循环（评估模型）
    model.eval()  # 设置为评估模式（禁用 dropout 等）
    correct = 0
    total = 0
    with torch.no_grad():  # 无梯度计算，节省内存
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大 logit 的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

print("Training completed!")