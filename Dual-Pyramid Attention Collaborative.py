import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import random
from tqdm import tqdm  # 导入 tqdm
import cv2
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x

class SubNetwork(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, mlp_dim, num_layers):
        super(SubNetwork, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        # Flatten spatial dimensions into sequences
        self.flatten = nn.Flatten(2)  # Convert (H, W) -> (H*W)
        self.embedding = nn.Linear(input_size**2, embed_dim)  # Embedding for input

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 展平空间维度并传递到 embedding 层
        x = x.view(x.size(0), -1)  # [B, H*W]
        x = self.embedding(x)  # [B, embed_dim]

        # 添加 batch 维度（因为 MultiheadAttention 需要 [seq_len, batch_size, embed_dim]）
        x = x.unsqueeze(1)  # [B, 1, embed_dim]

        # Transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x)

        # Global average pooling
        x = x.permute(1, 2, 0)  # [B, embed_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [B, embed_dim]
        return x

class MultiScaleNetwork(nn.Module):
    def __init__(self, num_classes, embed_dims, num_heads, mlp_dims, num_layers):
        super(MultiScaleNetwork, self).__init__()
        # Three sub-networks for different input sizes
        self.subnet1 = SubNetwork(56, embed_dims[0], num_heads[0], mlp_dims[0], num_layers[0])
        self.subnet2 = SubNetwork(28, embed_dims[1], num_heads[1], mlp_dims[1], num_layers[1])
        self.subnet3 = SubNetwork(14, embed_dims[2], num_heads[2], mlp_dims[2], num_layers[2])

        # Fully connected layer for classification
        self.fc = nn.Linear(sum(embed_dims), num_classes)

    def forward(self, x1, x2, x3):
        # Process each input with its corresponding sub-network
        out1 = self.subnet1(x1)
        out2 = self.subnet2(x2)
        out3 = self.subnet3(x3)

        # Concatenate the outputs
        out = torch.cat([out1, out2, out3], dim=-1)

        # Fully connected layer
        out = self.fc(out)
        return out


# 骨架提取函数
def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): 图像文件路径列表。
            labels (list): 对应的数字标签列表。
            transform (callable, optional): 图像预处理操作。
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # 加载灰度图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            # 提取骨架特征并调整尺寸
            skeleton = skeletonize(image)
            skeleton_56 = resize(skeleton, (56, 56), mode='constant', anti_aliasing=True).astype(np.float32)
            skeleton_28 = resize(skeleton, (28, 28), mode='constant', anti_aliasing=True).astype(np.float32)
            skeleton_14 = resize(skeleton, (14, 14), mode='constant', anti_aliasing=True).astype(np.float32)

            # 加载 RGB 图像
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # 返回数据
            return image, skeleton_56, skeleton_28, skeleton_14, label

        except Exception as e:
            print(f"错误加载图像 {image_path}: {e}")
            return None
def visualize_sca1(SCA1, batch_index=0, num_features=8):
    """
    可视化 SCA1 中的特征图。

    Args:
        SCA1 (torch.Tensor): 输入特征图，形状为 (batch, channels, height, width)。
        batch_index (int): 要可视化的 batch 索引。
        num_features (int): 可视化的特征图数量。
    """
    # 确保输入是 Tensor
    assert isinstance(SCA1, torch.Tensor), "SCA1 必须是一个 PyTorch Tensor！"
    batch_size, channels, height, width = SCA1.shape

    # 检查索引合法性
    assert batch_index < batch_size, f"batch_index 超出范围！最大值为 {batch_size - 1}"
    assert num_features <= channels, f"num_features 不能大于总通道数 {channels}"

    # 获取指定 Batch 的特征图
    features = SCA1[batch_index]  # (channels, height, width)

    # 选择要可视化的特征图
    selected_features = features[:num_features]  # 取前 num_features 个通道

    # 设置画布
    cols = 4  # 每行显示的特征图数量
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    # 可视化每个特征图
    for i in range(num_features):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        ax.imshow(selected_features[i].detach().cpu().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Feature {i+1}")

    # 隐藏多余的子图
    for j in range(num_features, rows * cols):
        if rows > 1:
            axes[j // cols, j % cols].axis('off')
        else:
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# def spatial_attention_numpy(feature_map1, feature_map2):
#     """
#     计算两个特征图之间的空间注意力，输入为 NumPy 数组。
#
#     参数:
#     - feature_map1: ndarray, 形状为 (H, W)
#     - feature_map2: ndarray, 形状为 (H, W)
#
#     返回:
#     - output: ndarray, 加权后的特征图，形状为 (H, W)
#     """
#     # 将 NumPy 数组转换为 PyTorch Tensor
#     feature_map1 = torch.tensor(feature_map1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#     feature_map2 = torch.tensor(feature_map2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#
#     # 1. 计算相似性 (逐元素相乘)
#     similarity_map = feature_map1 * feature_map2  # [1, 1, H, W]
#
#     # 2. 归一化权重 (Softmax)
#     attention_weights = F.softmax(similarity_map.view(1, -1), dim=-1).view_as(similarity_map)  # [1, 1, H, W]
#
#     # 3. 加权融合
#     output = attention_weights * feature_map1  # [1, 1, H, W]
#
#     # 转回 NumPy 数组并去掉多余维度
#     return output.squeeze().numpy()
def spatial_attention(feature_map1, feature_map2):
    """
    计算两个特征图之间的空间注意力，输入为 PyTorch 张量。

    参数:
    - feature_map1: Tensor, 形状为 (H, W)，需在 GPU 上。
    - feature_map2: Tensor, 形状为 (H, W)，需在 GPU 上。

    返回:
    - output: Tensor, 加权后的特征图，形状为 (H, W)，在 GPU 上。
    """
    # 扩展维度以便进行计算
    feature_map1 = feature_map1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    feature_map2 = feature_map2.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # 1. 计算相似性 (逐元素相乘)
    similarity_map = feature_map1 * feature_map2  # [1, 1, H, W]

    # 2. 归一化权重 (Softmax)
    attention_weights = F.softmax(similarity_map.view(1, -1), dim=-1).view_as(similarity_map)  # [1, 1, H, W]

    # 3. 加权融合
    output = attention_weights * feature_map1  # [1, 1, H, W]

    # 去掉多余维度
    return output.squeeze(0).squeeze(0)  # [H, W]

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

# 数据预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 加载数据集并提取文件路径和标签
def load_dataset(root_dir):
    file_paths = []
    labels = []
    class_to_idx = {}
    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            class_to_idx[class_name] = idx
            for file_name in os.listdir(class_path):
                file_paths.append(os.path.join(class_path, file_name))
                labels.append(idx)
    return file_paths, labels


if __name__ == '__main__':

    # 数据集路径
    data_dir = r'D:\DataSet\output_small_dataset'  # 替换为实际数据集路径

    # 加载数据集
    file_paths, labels = load_dataset(data_dir)

    # 打乱数据
    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)

    # 划分数据集 (8:1:1)
    total_size = len(file_paths)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    train_files, train_labels = file_paths[:train_size], labels[:train_size]
    val_files, val_labels = file_paths[train_size:train_size + val_size], labels[train_size:train_size + val_size]
    test_files, test_labels = file_paths[train_size + val_size:], labels[train_size + val_size:]

    # 创建数据集
    train_dataset = CustomDataset(train_files, train_labels, transform=transform)
    val_dataset = CustomDataset(val_files, val_labels, transform=transform)
    test_dataset = CustomDataset(test_files, test_labels, transform=transform)

    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 输出分割信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")


    # 加载预训练 ResNet18
    ResNetmodel = models.resnet18(weights=True)
    for param in ResNetmodel.parameters():
        param.requires_grad = False

    # 提取 ResNet 中间特征
    layer1 = nn.Sequential(*list(ResNetmodel.children())[:5])  # [1, 64, 56, 56]
    layer2 = nn.Sequential(*list(ResNetmodel.children())[:6])  # [1, 128, 28, 28]
    layer3 = nn.Sequential(*list(ResNetmodel.children())[:7])  # [1, 256, 14, 14]



    # 初始化 MultiScaleNetwork
    num_classes = 183
    model = MultiScaleNetwork(num_classes, embed_dims=[128, 256, 512], num_heads=[4, 8, 16], mlp_dims=[256, 512, 1024],
                              num_layers=[2, 2, 2])
    # 将模型移动到 GPU
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10


    # 训练和验证
    for epoch in range(num_epochs):  # 假设训练 10 个 epoch
        model.train()
        epoch_loss = 0
        # 创建 tqdm 进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for images, skeleton_56, skeleton_28, skeleton_14, labels in train_loader:
                # 转为张量 turn to Tensor


                skeleton_56 = skeleton_56.unsqueeze(1).clone().detach().to(device)  # [B, 1, 56, 56]
                skeleton_28 = skeleton_28.unsqueeze(1).clone().detach().to(device)  # [B, 1, 28, 28]
                skeleton_14 = skeleton_14.unsqueeze(1).clone().detach().to(device)  # [B, 1, 14, 14]

                labels = labels.to(device)

                # 提取 ResNet feature
                f3 = layer1(images)  # [B, 64, 56, 56]
                f4 = layer2(images)  # [B, 128, 28, 28]
                f5 = layer3(images)  # [B, 256, 14, 14]

                f3_compressed = torch.mean(f3, dim=1, keepdim=True)
                f4_compressed = torch.mean(f4, dim=1, keepdim=True)
                f5_compressed = torch.mean(f5, dim=1, keepdim=True)

                f3_squeezed = f3_compressed.squeeze(1).to(device)
                f4_squeezed = f4_compressed.squeeze(1).to(device)
                f5_squeezed = f5_compressed.squeeze(1).to(device)

                # 空间注意力融合 SCA(spatial collaborative attention)

                SCA1 = torch.stack([spatial_attention(f3_squeezed[i], skeleton_56[i]) for i in range(f3_squeezed.size(0))]).to(device)
                SCA2 = torch.stack(
                    [spatial_attention(f4_squeezed[i], skeleton_28[i]) for i in range(f4_squeezed.size(0))]).to(device)
                SCA3 = torch.stack(
                    [spatial_attention(f5_squeezed[i], skeleton_14[i]) for i in range(f5_squeezed.size(0))]).to(device)

                # 前向传播Forward
                outputs = model(SCA1, SCA2, SCA3)
                loss = criterion(outputs, labels)

                # 反向传播与优化 Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 更新 tqdm 进度条
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")




