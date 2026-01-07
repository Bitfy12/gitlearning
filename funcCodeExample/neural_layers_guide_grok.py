import torch
import torch.nn as nn

# ======================================
# 神经网络中的特殊层介绍与示例（更新版）
# ======================================

# 使用说明：
# - 运行本文件时，会执行每个示例的 print 语句，展示输出结果。
# - 假设读者有基本的 PyTorch 知识；需安装 PyTorch (pip install torch)。
# - 示例使用随机或固定输入，便于理解；实际应用中替换为模型中的 forward 方法。
# - 注意：示例在 CPU 上运行；如需 GPU，使用 .to(device)。
# - 扩展重点：BatchNorm 变体在不同维度数据上的应用（如1D序列、2D图像、3D视频/体积数据），以及在训练中的稳定性提升。

# ======================================
# 1. 激活函数层 (Activation Layers)
# ======================================
# 激活函数引入非线性，帮助网络学习复杂模式。常见于隐藏层后。
# 扩展说明：这些层通常放置在线性层（如Conv或Linear）之后，帮助模型捕捉非线性关系。在训练中，选择合适的激活函数可以加速收敛、减少梯度问题，并提高泛化能力。例如，在计算机视觉任务中，ReLU 变体常用于处理像素级非线性；在 NLP 中，Tanh 用于序列建模以保持值域平衡。

# ----------------------
# nn.ReLU (Rectified Linear Unit)
# ----------------------
# 原理：输出 max(x, 0)，正值不变，负值置零。简单高效，避免梯度消失问题（相比 Sigmoid）。
# 参数：
#   - inplace: 布尔值，默认 False。如果 True，则原地操作节省内存，但可能覆盖输入。
# 适用场景：几乎所有隐藏层，尤其是 CNN 中的卷积层后（如 ResNet 架构，用于图像分类、目标检测），或 MLP 中的全连接隐藏层（如用于回归或分类任务）。在深度网络中，ReLU 加速收敛，因为其线性部分允许梯度直接流动；适用于计算机视觉（如 ImageNet 分类）、自然语言处理（如 BERT 的前馈层），或强化学习（如 DQN 的 Q-network）。益处：计算高效，减少过拟合风险，但需监控死神经元问题。在大型数据集训练中，能缩短训练时间 2-3 倍。
# 注意事项：可能导致“死神经元”（负值梯度为零）。在深层网络中，结合 BatchNorm 使用效果更好（如在 ResNet 中，顺序为 Conv -> BN -> ReLU）。

relu = nn.ReLU(inplace=True)
x_relu = torch.tensor([-1.0, 0.0, 1.0])
output_relu = relu(x_relu)
print("nn.ReLU 示例输出:", output_relu)  # 预期: tensor([0., 0., 1.])

# ----------------------
# nn.LeakyReLU (Leaky Rectified Linear Unit)
# ----------------------
# 原理：改进 ReLU，对于负值输出 negative_slope * x，允许小梯度流动，避免死神经元。
# 参数：
#   - negative_slope: 负值斜率，默认 0.01。
#   - inplace: 同上。
# 适用场景：当标准 ReLU 导致死神经元时使用，例如在生成对抗网络 (GAN) 中（如 DCGAN 的生成器和判别器，用于图像生成任务），或非常深的网络（如超过 100 层的 CNN，用于医疗图像分割）。在噪声数据或梯度稀疏的任务中（如语音识别或时间序列预测），LeakyReLU 允许负梯度流动，提高训练稳定性。益处：相比 ReLU，减少了模型在低激活区域的失效风险，在实践中可提升 5-10% 的准确率，尤其在不平衡数据集上。
# 注意事项：斜率过大可能引入噪声；通常与 Dropout 结合（如在 GAN 中，顺序为 Conv -> LeakyReLU -> Dropout）。

leaky_relu = nn.LeakyReLU(negative_slope=0.1)
x_leaky = torch.tensor([-1.0, 1.0])
output_leaky = leaky_relu(x_leaky)
print("nn.LeakyReLU 示例输出:", output_leaky)  # 预期: tensor([-0.1000,  1.0000])

# ----------------------
# nn.Sigmoid
# ----------------------
# 原理：输出 1 / (1 + exp(-x))，范围 [0,1]。
# 参数：无主要参数。
# 适用场景：主要用于二分类输出层（如逻辑回归模型，或在旧式神经网络中用于门控机制，如 LSTM 的遗忘门）。在医疗诊断（如二元分类肿瘤良恶性）或推荐系统（如点击率预测）中，作为最终输出以产生概率。扩展：在浅层网络或需要严格 [0,1] 范围的任务中有效，但不适合隐藏层，因为梯度消失会导致深层网络训练困难（如在多层感知机中，如果深度超过 5 层，梯度可能衰减到 10^-5 以下）。
# 注意事项：在深层网络中避免使用；输出层时，与 BCE 损失搭配（如在二分类任务中，Sigmoid + BCELoss 组合）。

sigmoid = nn.Sigmoid()
x_sigmoid = torch.tensor([-1.0, 0.0, 1.0])
output_sigmoid = sigmoid(x_sigmoid)
print("nn.Sigmoid 示例输出:", output_sigmoid)  # 预期: tensor([0.2689, 0.5000, 0.7311]) (约值)

# ----------------------
# nn.Tanh (Hyperbolic Tangent)
# ----------------------
# 原理：输出 (exp(x) - exp(-x)) / (exp(x) + exp(-x))，范围 [-1,1]。
# 参数：无。
# 适用场景：常用于 RNN/LSTM 的隐藏层（如在序列建模中，用于保持输出平衡，例如机器翻译或情感分析任务）。在零中心数据处理中有效，因为输出对称于零，能减少偏置。在时间序列预测（如股票价格）或语音合成（如 WaveNet）中，Tanh 帮助稳定梯度流动。扩展：在循环网络中，如果使用 Sigmoid 会导致梯度爆炸，Tanh 通过 [-1,1] 范围限制输出，提高数值稳定性，尤其在长序列（如 100+ 时间步）任务中。
# 注意事项：类似 Sigmoid，易梯度消失；现代网络中被 ReLU 取代，但在某些序列模型中仍有用（如 GRU 的更新门）。

tanh = nn.Tanh()
x_tanh = torch.tensor([-1.0, 0.0, 1.0])
output_tanh = tanh(x_tanh)
print("nn.Tanh 示例输出:", output_tanh)  # 预期: tensor([-0.7616,  0.0000,  0.7616]) (约值)

# ======================================
# 2. 正则化层 (Regularization Layers)
# ======================================
# 这些层防止过拟合，提高泛化能力。扩展：正则化层在训练大模型时至关重要，能减少内部协变量偏移（ICS），使模型对 batch_size 和学习率更鲁棒。在实际项目中，常用于数据增强不足的场景，如小数据集训练。

# ----------------------
# nn.Dropout
# ----------------------
# 原理：训练时随机“丢弃”部分神经元（置零），迫使网络学习冗余表示。推理时全使用，但缩放输出。
# 参数：
#   - p: 丢弃概率，默认 0.5。
#   - inplace: 同上。
# 适用场景：放置在全连接层后或卷积层后（如在 AlexNet 或 VGG 中，用于图像分类任务），防止过拟合。扩展：在计算机视觉中，用于大型 CNN 的分类头（如在 ImageNet 数据集上，Dropout p=0.5 可减少 10-20% 的过拟合）；在 NLP 中，用于 Transformer 的前馈层（如 BERT 的 dropout，处理长文本时防止参数过度依赖）。益处：在小数据集（如几千样本）训练中，提升泛化 5-15%；在 ensemble 效果模拟中，相当于训练多个子网络。
# 注意事项：在 model.train() 时生效，model.eval() 时关闭。不要用于输入层或输出层。结合 BatchNorm 使用时，顺序重要（通常 BatchNorm -> 激活 -> Dropout）。

dropout = nn.Dropout(p=0.5)
x_dropout = torch.ones(1, 5)  # 示例输入
output_dropout = dropout(x_dropout)
print("nn.Dropout 示例输出 (训练模式，随机):", output_dropout)  # 预期: 类似 tensor([[0., 2., 0., 2., 0.]])

# ----------------------
# nn.BatchNorm1d (Batch Normalization for 1D data)
# ----------------------
# 原理：对每个 batch 的特征进行归一化（均值 0，方差 1），然后缩放/偏移。减少内部协变量偏移，加速训练。
# 参数：
#   - num_features: 特征通道数。
#   - eps: 防止除零，默认 1e-5。
#   - momentum: 运行均值/方差的动量，默认 0.1。
#   - affine: 是否学习缩放/偏移参数，默认 True。
# 适用场景：用于 1D 数据，如 MLP 中的全连接层后（例如在 tabular 数据分类任务中，如 Kaggle 竞赛的结构化数据处理），或序列模型如 RNN 的隐藏状态（在时间序列预测中，如股票数据分析）。扩展：在 batch_size 较大的训练中（如 32+），BatchNorm1d 稳定梯度，允许更高学习率（e.g., 0.01 而非 0.001），缩短训练时间；在不平衡数据集上，减少分布偏移，提高准确率 3-10%。在强化学习代理（如 PPO）中，用于规范化状态特征。
# 注意事项：batch_size 不能太小（至少 8+，否则统计不准）。在 model.eval() 时使用运行统计。顺序通常为 Linear -> BatchNorm1d -> ReLU。

bn1d = nn.BatchNorm1d(num_features=3)
x_bn1d = torch.randn(2, 3)  # batch_size=2, features=3
output_bn1d = bn1d(x_bn1d)
print("nn.BatchNorm1d 示例输出:", output_bn1d)  # 预期: 归一化结果，接近均值0，方差1

# ----------------------
# nn.BatchNorm2d (Batch Normalization for 2D data)
# ----------------------
# 原理：同上，但针对 2D 数据（如图像），对每个通道的 HxW 进行归一化。
# 参数：同 nn.BatchNorm1d，但 num_features 对应通道数。
# 适用场景：主要用于 CNN 中的卷积层后（如 ResNet 或 U-Net，用于图像分类、分割或检测任务）。扩展：在计算机视觉中，如 COCO 数据集的目标检测，BatchNorm2d 规范化特征图，减少训练时的分布变化，允许更深网络（e.g., 50+ 层）而无梯度爆炸；在转移学习中（如 fine-tune pretrained model），它保持特征一致性，提高下游任务准确率 5-15%。在实时应用如自动驾驶中，加速收敛以减少计算资源。
# 注意事项：输入形状为 (N, C, H, W)。与 Conv2d 结合时，顺序为 Conv2d -> BatchNorm2d -> ReLU。batch_size 过小会导致噪声统计。

bn2d = nn.BatchNorm2d(num_features=1)  # 示例：1 通道
x_bn2d = torch.randn(2, 1, 4, 4)  # batch_size=2, channels=1, height=4, width=4
output_bn2d = bn2d(x_bn2d)
print("nn.BatchNorm2d 示例输出形状:", output_bn2d.shape)  # 预期: torch.Size([2, 1, 4, 4])，值归一化

# ----------------------
# nn.BatchNorm3d (Batch Normalization for 3D data)
# ----------------------
# 原理：同上，但针对 3D 数据（如视频或体积数据），对每个通道的 DxHxW 进行归一化。
# 参数：同上。
# 适用场景：用于 3D CNN 中，如视频分类（e.g., Kinetics 数据集的动作识别）或医疗成像（如 CT 扫描的 3D 体积分割）。扩展：在时空序列任务中，如视频理解模型（e.g., 3D ResNet），BatchNorm3d 规范化深度维度特征，处理时间相关偏移，提高时序一致性；在生物医学中，如 MRI 图像分析，减少扫描噪声影响，提升分割精度 10-20%。益处：在高维数据上稳定训练，允许更大 batch_size 以利用 GPU 内存。
# 注意事项：输入形状为 (N, C, D, H, W)。计算开销较高，适合 GPU 训练。batch_size 需足够大以获取可靠统计。

bn3d = nn.BatchNorm3d(num_features=1)  # 示例：1 通道
x_bn3d = torch.randn(2, 1, 3, 4, 4)  # batch_size=2, channels=1, depth=3, height=4, width=4
output_bn3d = bn3d(x_bn3d)
print("nn.BatchNorm3d 示例输出形状:", output_bn3d.shape)  # 预期: torch.Size([2, 1, 3, 4, 4])，值归一化

# ----------------------
# nn.LayerNorm (Layer Normalization)
# ----------------------
# 原理：对每个样本的特征进行归一化（不依赖 batch），常用于 Transformer。
# 参数：
#   - normalized_shape: 要归一化的形状（如 [hidden_size]）。
#   - eps: 同上。
# 适用场景：序列模型如 Transformer 中的注意力层后（e.g., GPT 或 BERT，用于机器翻译或文本生成任务）。扩展：在 NLP 中，如处理变长句子，LayerNorm 不受 batch_size 影响，稳定梯度在长序列（e.g., 512+ tokens）中流动；在小 batch 训练（如分布式训练的 per-device batch=1）中，更优于 BatchNorm，提高模型鲁棒性。益处：在自注意力机制中，减少层间分布偏移，提升训练速度和最终 BLEU 分数。
# 注意事项：比 BatchNorm 更稳定，但计算稍慢。常用于 [batch, seq_len, features] 形状。

ln = nn.LayerNorm(normalized_shape=3)
x_ln = torch.randn(2, 3)
output_ln = ln(x_ln)
print("nn.LayerNorm 示例输出:", output_ln)  # 预期: 每个样本独立归一化，均值0，方差1

# ======================================
# 3. 池化层 (Pooling Layers)
# ======================================
# 主要用于 CNN，降维并提取显著特征。输入通常为 4D 张量 (batch, channels, height, width)。
# 扩展：池化层在特征提取中关键，用于下采样以减少参数和计算量。在多尺度任务中，如目标检测的特征金字塔网络 (FPN)，池化帮助融合不同分辨率特征。

# ----------------------
# nn.MaxPool2d (Maximum Pooling)
# ----------------------
# 原理：在 kernel 内取最大值，保留强特征。
# 参数：
#   - kernel_size: 池化窗口大小，如 2 或 (2,2)。
#   - stride: 步长，默认等于 kernel_size。
#   - padding: 填充，默认 0。
#   - dilation: 膨胀，默认 1。
# 适用场景：图像分类/检测中的下采样层（如 VGG 或 Faster R-CNN，用于提取边缘、纹理等显著特征）。扩展：在物体检测任务中（如 YOLO），MaxPool2d 保留高响应区域，帮助定位小物体；在语义分割（如 FCN），用于多尺度特征聚合，处理不同大小的目标。益处：鲁棒于噪声，减少过拟合，在高分辨率图像（e.g., 1024x1024）上将维度减半，节省 75% 计算。
# 注意事项：可能丢失细节；用于下采样，减少参数。padding 可保持边界信息。

maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
x_maxpool = torch.tensor([[[[1,2],[3,4]]]])  # (1,1,2,2)
output_maxpool = maxpool(x_maxpool.float())  # 转为 float 以兼容
print("nn.MaxPool2d 示例输出:", output_maxpool)  # 预期: tensor([[[[4]]]])

# ----------------------
# nn.AvgPool2d (Average Pooling)
# ----------------------
# 原理：在 kernel 内取平均值，平滑特征。
# 参数：同 MaxPool2d。
# 适用场景：全局平均池化（GAP）在分类头中（如 MobileNet 或 EfficientNet，用于移动设备图像分类），或平滑噪声任务（如低质量图像增强）。扩展：在轻量级模型中，AvgPool2d 替换全连接层，减少参数（e.g., 从百万到千级），适合边缘计算；在 GAN 中，用于判别器平滑特征图，提高生成质量。益处：比 MaxPool 更注重整体分布，在模糊图像或噪声数据上提升鲁棒性 5-10%。
# 注意事项：比 MaxPool 更平滑，但可能弱化强特征。常用于最终层以产生固定大小输出。

avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
x_avgpool = torch.tensor([[[[1,2],[3,4]]]])  # (1,1,2,2)
output_avgpool = avgpool(x_avgpool.float())
print("nn.AvgPool2d 示例输出:", output_avgpool)  # 预期: tensor([[[[2.5]]]])

# ======================================
# 4. 其他相关特殊层 (Other Related Layers)
# ======================================
# 这些层辅助网络结构，如展平或概率转换。扩展：这些层在网络过渡中不可或缺，例如在混合架构（如 CNN+RNN）中用于维度转换。

# ----------------------
# nn.Flatten
# ----------------------
# 原理：展平张量，如从 (batch, C, H, W) 到 (batch, C*H*W)。
# 参数：
#   - start_dim: 起始维度，默认 1。
#   - end_dim: 结束维度，默认 -1。
# 适用场景：CNN 到全连接层的过渡（如在 LeNet 或 ResNet 的分类头，用于将特征图转换为向量输入分类器）。扩展：在多模态任务中（如图像+文本），Flatten 处理图像特征后与序列特征拼接；在迁移学习中，用于冻结骨干网络后添加自定义头。益处：简化维度管理，在高维输入（如 3D 扫描）上避免手动 reshape 错误。
# 注意事项：无参数学习，常用于 forward 中。确保 start_dim 避免展平 batch 维度。

flatten = nn.Flatten()
x_flatten = torch.randn(1, 3, 2, 2)
output_flatten = flatten(x_flatten)
print("nn.Flatten 示例输出形状:", output_flatten.shape)  # 预期: torch.Size([1, 12])

# ----------------------
# nn.Softmax
# ----------------------
# 原理：将 logits 转换为概率，sum=1。
# 参数：
#   - dim: softmax 维度，默认 -1。
# 适用场景：多分类输出层（如在 MNIST 手写数字识别，或 ImageNet 1000 类分类）。扩展：在概率输出任务中，如情感分析（多标签），Softmax 产生互斥概率；在注意力机制（如 Transformer 的 scaled dot-product），用于权重归一化。益处：与 CrossEntropyLoss 结合时，避免数值不稳定；在不确定性估计中，可用于置信度计算。
# 注意事项：不要在损失前手动加，除非需要概率（如在推理时输出 top-k 类别）。

softmax = nn.Softmax(dim=1)
x_softmax = torch.tensor([[1.0, 2.0]])
output_softmax = softmax(x_softmax)
print("nn.Softmax 示例输出:", output_softmax)  # 预期: tensor([[0.2689, 0.7311]]) (约值)

# ======================================
# 总结与使用建议
# ======================================
# - 组合使用示例：在 CNN 中顺序：Conv -> BatchNorm (1d/2d/3d 根据数据) -> ReLU -> MaxPool -> Dropout。这在图像任务中稳定训练，减少过拟合。
# - 训练细节：在 model.train() 和 model.eval() 间切换（影响 Dropout/BN）。监控过拟合时增加 Dropout p 值，或使用 TensorBoard 跟踪激活分布。
# - 性能影响：激活如 ReLU 加速训练（减少计算）；正则化如 BN/Dropout 提升泛化（在小数据集上显著）；池化减少计算量（在移动设备上关键）。
# - BatchNorm 扩展：1d 适合 tabular/序列数据（如金融时间序列分析）；2d 适合 2D 图像（如卫星图像分类）；3d 适合体积/视频数据（如 3D 打印缺陷检测或视频动作识别）。
# - 扩展：探索更多层如 nn.ELU, nn.GELU (用于 Transformer) 或 nn.Conv2d (卷积层)。
# - 如果运行本文件，观察 print 输出以验证示例。