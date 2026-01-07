import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


def test_randn():
    t = torch.randn(2, 3)   # 2x3 矩阵，均值0方差1
    n = np.random.randn(2, 3)
    print("torch.randn:\n", t, "\nshape:", t.shape)
    print("np.random.randn:\n", n, "\nshape:", n.shape)

def test_arange():
    t = torch.arange(10)   # [0,1,2,3,4,5,6,7,8,9]
    n = np.arange(0, 10, 2) # [0,2,4,6,8]
    print("torch.arange:", t, "shape:", t.shape)
    print("n.arange:", n, "shape:", n.shape)

def test_concatenate():
    a = torch.tensor([[1,2],[3,4]])
    b = torch.tensor([[5,6]])
    t = torch.cat([a, b], dim=0)   # 在行方向拼接
    n = np.concatenate([a.numpy(), b.numpy()], axis=0)
    print("torch.cat:\n", t, "shape:", t.shape)
    print("np.concatenate:\n", n, "shape:", n.shape)

def test_dim_concat_channels():
    x1 = torch.randn(2, 16, 32, 32)  # (batch=2, channel=16, H=32, W=32)
    x2 = torch.randn(2, 16, 32, 32)
    y = torch.cat([x1, x2], dim=1)  # 在通道维度拼接
    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)
    print("concat (dim=1):", y.shape)  # (2, 32, 32, 32)

def test_dim_softmax():
    logits = torch.randn(2, 5)  # (batch=2, classes=5)
    probs = F.softmax(logits, dim=1)  # 在类别维度 softmax
    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("sum over classes:", probs.sum(dim=1))  # 每行加起来=1

def test_dim_spatial_pooling():
    x = torch.randn(2, 64, 8, 8)   # (batch=2, channel=64, H=8, W=8)
    pooled = x.mean(dim=(2, 3))    # (2, 64), 每个通道取空间平均
    print("x shape:", x.shape)
    print("pooled shape:", pooled.shape)

def test_argmax():
    a = torch.tensor([[1,5,3],[7,2,9]])
    t = torch.argmax(a, dim=1)   # 每行最大值索引
    n = np.argmax(a.numpy(), axis=1)
    print("tensor:\n", a)
    print("torch.argmax:", t, "shape:", t.shape)
    print("np.argmax:", n, "shape:", n.shape)

def test_stack():
    a = torch.tensor([1,2,3])
    b = torch.tensor([4,5,6])
    t = torch.stack([a, b], dim=0)   # 新维度 0
    n = np.stack([a.numpy(), b.numpy()], axis=0)
    print("torch.stack:\n", t, "shape:", t.shape)
    print("np.stack:\n", n, "shape:", n.shape)

def test_permute():
    a = torch.randn(2,3,4)   # shape: (2,3,4)
    t = a.permute(1,0,2)     # 交换前两维 (3,2,4)
    n = np.transpose(a.numpy(), (1,0,2))
    print("original shape:", a.shape)
    print("torch.permute shape:", t.shape)
    print("np.transpose shape:", n.shape)

def test_reshape():
    a = torch.arange(6)   # [0,1,2,3,4,5]
    t = a.reshape(2,3)    # 变形
    n = a.numpy().reshape(2,3)
    print("reshape torch:\n", t, "shape:", t.shape)
    print("reshape numpy:\n", n, "shape:", n.shape)


def test_unsqueeze_squeeze():
    '''
    #增加/去掉维度
    #unsqueeze(dim) 增加一维，dim指定位置

    #numpy.squeeze(a, axis=None) 移除数组 a 中所有维度为1的轴（单维轴）。
    #如果指定了 axis 参数，则只移除指定轴上维度为1的轴。
    #如果指定的某个轴的维度不为1，则会报错。

    #PyTorch 的 squeeze(dim)：如果指定的 dim 维度不是1，不会抛出错误，只是返回原张量不变。  
    '''
    a = torch.tensor([[1,2,3], [4,5,6]])        # shape: (3,)
    t = a.unsqueeze(1)               # 增加一维 (1,3)
    s = t.squeeze(0)                 # 去掉指定位置的维度大小为1的维度，若该维度不为1则不变
    print("unsqueeze shape:", t.shape)
    print("squeeze shape:", s.shape)
    print("asdasd:", torch.squeeze(a, dim=0).shape)  # (2,3)，dim=0维度是2，不是1，不变


def test_linspace():
    t = torch.linspace(0, 1, steps=5)   # [0,0.25,0.5,0.75,1]
    n = np.linspace(0, 1, num=5)
    print("torch.linspace:", t)
    print("np.linspace:", n)

def test_freeze_parameters():
    # 一个简单的模型：特征提取部分 + 分类器部分
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 特征提取
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*8*8, 10)   # 分类器
    )
    
    # 假设输入图像是 (3,8,8)
    x = torch.randn(4, 3, 8, 8)  # batch=4
    y = torch.randint(0, 10, (4,))  # 随机标签
    
    # 冻结前几层 (只训练分类器部分)
    for param in model[0].parameters():   # Conv2d 层的参数
        param.requires_grad = False
    
    # 定义优化器，只会更新未冻结的参数
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # 训练一个小步骤
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 检查哪些参数被更新
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

def test_conv2d():
    x = torch.randn(1, 3, 32, 32)   # batch=1, channels=3, 32x32 图像
    conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
    y = conv(x)
    print("input shape:", x.shape)    # (1,3,32,32)
    print("output shape:", y.shape)   # (1,6,32,32) 因为 padding=1 保持大小不变


def test_sequential():
    # nn.Sequential 是一种序列容器，相当于一个字典，存放了我们需要的东西，
    # 这里存放的东西一般为神经网络基本元素，如Relu，Linear等
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    x = torch.randn(4, 10)   # batch=4, features=10
    y = model(x)
    print("input shape:", x.shape)     # (4,10)
    print("output shape:", y.shape)    # (4,5)

def test_save_load():
    model = nn.Linear(10, 2)
    x = torch.randn(1, 10)
    y1 = model(x)

    # 保存参数
    torch.save(model.state_dict(), "model.pth")

    # 新建一个同结构的模型
    model2 = nn.Linear(10, 2)
    model2.load_state_dict(torch.load("model.pth"))

    y2 = model2(x)
    print("Output before save:", y1)
    print("Output after load:", y2)   # 和之前一致


def test_relu():
    x = torch.tensor([[-1.0, 0.0, 2.0]])
    relu_layer = nn.ReLU()
    y1 = relu_layer(x)
    y2 = F.relu(x)
    print("input:", x)
    print("nn.ReLU:", y1)
    print("F.relu:", y2)

def test_loss():
    pred = torch.tensor([[2.0, 1.0, 0.1]])   # logits
    target = torch.tensor([0])               # 真实类别为 class 0
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred, target)
    print("pred shape:", pred.shape)
    print("target:", target)
    print("loss:", loss.item())

def test_optimizer():
    model = nn.Linear(2, 1)

    '''
    SGD (随机梯度下降)
    参数:
    params: 待优化的神经网络参数
    lr: 学习率
    momentum: 动量
    weight_decay: 权重衰减
    nesterov: 是否使用nesterov动量
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0]])

    for step in range(3):
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step}, loss {loss.item()}")

def test_dropout():
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(10)
    dropout.train()   # 开启训练模式
    print("train mode:", dropout(x))   # 有些会变成0
    dropout.eval()    # 评估模式
    print("eval mode:", dropout(x))    # 不会丢弃

def test_batchnorm():
    bn = nn.BatchNorm2d(3)    # 对3个通道做BN
    x = torch.randn(2, 3, 4, 4)   # batch=2, channels=3
    y = bn(x)
    print("input shape:", x.shape)
    print("output shape:", y.shape)

def slice_batch():
    # 切分batch
    x = torch.randn(8, 3, 32, 32)  # batch=8, channel=3, H=32, W=32
    sub_batch = x[:4]              # 取前四个，等价于 x[:4, :, :, :]，多重向量重点是有没有等号，没有等号就是默认第一个维度的切片操作“：”
    print("original:", x.shape)    # (8,3,32,32)
    print("sub_batch:", sub_batch.shape)  # (4,3,32,32)

def slice_channels():
    x = torch.randn(2, 64, 16, 16)  # batch=2, 64通道
    c1 = x[:, 0, :, :]              # 取第0个通道
    c_range = x[:, 0:16, :, :]      # 取前16个通道
    print("c1 shape:", c1.shape)        # (2,16,16)
    print("c_range shape:", c_range.shape)  # (2,16,16,16)

def slice_stride():
    # 步长切片，模拟下采样
    x = torch.randn(1, 3, 8, 8)
    y = x[:, :, ::2, ::2]   # 每隔2取一个 => 下采样
    print("original:", x.shape)  # (1,3,8,8)
    print("stride slice:", y.shape)  # (1,3,4,4)

def slice_flatten():
    # flatten操作，只适用于连续张量
    x = torch.randn(2, 3, 4, 4)   # batch=2, channel=3, 4x4
    y = x.view(2, 3, -1)          # 把 HxW 展平成 16 ， -1表示自动计算（省略的地方自动相乘）
    z = x.view(2, -1)             # -1 表示自动计算
    print("original:", x.shape)   # (2,3,4,4)
    print("flatten:", y.shape)    # (2,3,16)
    print("flattened:", z.shape)  # (2, 48)  => 3*4*4=48
    
def test_view_merge():
    x = torch.randn(4, 5, 6)   # (batch=4, seq_len=5, hidden=6)
    y = x.view(20, 6)          # 合并前两个维度
    print("original shape:", x.shape)   # (4,5,6)
    print("merged shape:", y.shape)     # (20,6)

def test_view_nn_usage():
    # 神经网络常用view操作
    # (1) 卷积输出 → 全连接
    x = torch.randn(32, 64, 7, 7)     # batch=32
    y = x.view(32, -1)
    print("conv2fc shape:", y.shape)  # (32,3136)

    # (2) Transformer: 拆 heads
    z = torch.randn(2, 5, 12*8)       # (batch, seq_len, hidden=96)
    z = z.view(2, 5, 12, 8)           # (batch, seq_len, heads=12, dim=8)
    print("transformer split shape:", z.shape)




if __name__ == '__main__':
    test_save_load()




