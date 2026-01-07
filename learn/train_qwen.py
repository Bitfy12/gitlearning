import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 数据集准备
def prepare_data():
    """
    数据集准备函数
    """
    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=10),    # 随机旋转
        transforms.ToTensor(),                    # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10的均值和标准差
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,           # 批次大小
        shuffle=True,            # 训练时通常需要打乱数据
        num_workers=4,           # 多进程加载数据
        pin_memory=True          # 如果GPU可用，加速数据传输
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,           # 测试时通常不需要打乱
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

# 2. 神经网络定义
class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # CIFAR-10图像大小为32x32，经过3次池化后变为4x4
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 卷积层 + 激活 + 池化
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 128*4*4]
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 3. 训练函数
def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """
    训练一个epoch的函数
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备（CPU/GPU）
        scheduler: 学习率调度器（可选）
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(pbar):
        # 将数据移动到指定设备
        data, target = data.to(device), target.to(device)
        
        # 4. 清零梯度
        optimizer.zero_grad()  # 重要：在每次反向传播前清零梯度
        
        # 5. 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 6. 反向传播
        loss.backward()
        
        # ✅ 重要注意事项：梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 7. 更新参数
        optimizer.step()
        
        # ✅ 重要注意事项：如果使用batch-wise的学习率调度器，在optimizer.step()之后调用
        if scheduler is not None and hasattr(scheduler, 'step') and not hasattr(scheduler, '_step_interval'):
            scheduler.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# 4. 验证函数
def validate(model, val_loader, criterion, device):
    """
    验证函数
    """
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 在验证时不计算梯度，节省内存
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# 5. 早停机制
class EarlyStopping:
    """
    早停机制类
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_model_state)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict().copy()

# 6. 主训练函数
def main():
    """
    主训练函数
    """
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据准备
    print("Loading data...")
    train_loader, test_loader = prepare_data()
    
    # 模型初始化
    print("Initializing model...")
    model = SimpleCNN(num_classes=10).to(device)
    
    # ✅ 重要注意事项：参数初始化（可选但推荐）
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001,              # 初始学习率
        weight_decay=1e-4      # L2正则化
    )
    
    # ✅ 重要注意事项：学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch学习率乘以0.1
    
    # 早停机制
    early_stopping = EarlyStopping(patience=10)
    
    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler  # 注意：这里传递scheduler
        )
        
        # 验证阶段
        val_loss, val_acc = validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # ✅ 重要注意事项：对于epoch-wise的学习率调度器，在验证后调用
        # 注意：StepLR是epoch-wise的，所以在这里调用
        if epoch % scheduler.step_size == 0:  # 或者简单地在每个epoch后调用
            scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_model.pth')
            print(f'✅ Best model saved with validation accuracy: {val_acc:.2f}%')
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # 绘制训练曲线
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练历史曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 模型测试函数
def test_model():
    """
    测试训练好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载最佳模型
    model = SimpleCNN(num_classes=10).to(device)
    
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 重新加载测试数据
    _, test_loader = prepare_data()
    
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 按类别统计准确率
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    # 打印各类别准确率
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

if __name__ == "__main__":
    # 运行主训练函数
    trained_model = main()
    
    # 测试模型
    print("\nTesting the trained model...")
    test_model()