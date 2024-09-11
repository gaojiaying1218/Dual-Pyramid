# This is main py that can train JiDaTop10 by Custom ViT model.
# The learning rate has to be 0.000001.
# Load this model = ViT(). which can viulize the attetnion by Visulizer_package.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
import timm
import time
from PIL import Image, ImageDraw
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# augmentations = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
#     transforms.RandomRotation(degrees=10),  # Randomly rotate the image by up to 10 degrees
#     transforms.Resize((256, 256)),  # Resize image to 256x256
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
# ])

#Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Load the dataset
dataset = ImageFolder(root=r'D:\DataSet\JiDaTop10/', transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% training, 20% testing
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        # Define your attention mechanisms here
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.key(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.value(x).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        x = attn_output.transpose(1, 2).reshape(B, N, -1)
        x = self.fc(x)
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim)
            ]))

    def forward(self, x):
        for norm1, attention, norm2, mlp1, activation, mlp2 in self.layers:
            x = norm1(x)
            x = attention(x)
            x += x
            x = norm2(x)
            x = mlp1(x)
            x = activation(x)
            x = mlp2(x)
            x += x
        return x

class CustomViT(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072): #image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072
        super(CustomViT, self).__init__()

        # Load pre-trained ViT model from timm
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the head (classifier) to match the number of classes in your dataset
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

        # Define other components of the ViT architecture
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # Assuming RGB images
        self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_encoder = TransformerEncoder(dim, depth, heads, mlp_dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average pool the sequence
        return self.mlp_head(x)

# Create an instance of your custom ViT model
num_classes = 10


# Example number of classes in your dataset
model = CustomViT(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001) #lr=0.000001 has the best performance
total_params_orignal = sum(p.numel() for p in model.parameters())
print(f"Total Parameters Orignal: {total_params_orignal}")
# Train the model
start_time = time.time()
# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Print training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Test the model
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Print test accuracy
    test_accuracy = 100 * correct_test / total_test
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")

# Save the model
end_time = time.time()
training_time = end_time - start_time
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")
print(f"Total training time: {training_time:.2f} seconds")
torch.save(model.state_dict(), 'CustomViT_timm_small.pth')

#0.0000001 result following:
# Using device: cuda
# Epoch 1/50, Train Loss: 2.3244, Train Accuracy: 11.72%
# Epoch 1/50, Test Accuracy: 14.76%
# Epoch 2/50, Train Loss: 2.2586, Train Accuracy: 17.80%
# Epoch 2/50, Test Accuracy: 15.34%
# Epoch 3/50, Train Loss: 2.2262, Train Accuracy: 18.14%
# Epoch 3/50, Test Accuracy: 15.53%
# Epoch 4/50, Train Loss: 2.2033, Train Accuracy: 20.09%
# Epoch 4/50, Test Accuracy: 25.24%
# Epoch 5/50, Train Loss: 2.1821, Train Accuracy: 24.56%
# Epoch 5/50, Test Accuracy: 24.47%
# Epoch 6/50, Train Loss: 2.1602, Train Accuracy: 24.95%
# Epoch 6/50, Test Accuracy: 24.85%
# Epoch 7/50, Train Loss: 2.1449, Train Accuracy: 24.85%
# Epoch 7/50, Test Accuracy: 25.63%
# Epoch 8/50, Train Loss: 2.1266, Train Accuracy: 25.88%
# Epoch 8/50, Test Accuracy: 25.44%
# Epoch 9/50, Train Loss: 2.1061, Train Accuracy: 25.88%
# Epoch 9/50, Test Accuracy: 24.85%
# Epoch 10/50, Train Loss: 2.0936, Train Accuracy: 26.31%
# Epoch 10/50, Test Accuracy: 24.27%
# Epoch 11/50, Train Loss: 2.0791, Train Accuracy: 27.58%
# Epoch 11/50, Test Accuracy: 24.47%
# Epoch 12/50, Train Loss: 2.0639, Train Accuracy: 27.29%
# Epoch 12/50, Test Accuracy: 24.27%
# Epoch 13/50, Train Loss: 2.0483, Train Accuracy: 28.60%
# Epoch 13/50, Test Accuracy: 25.63%
# Epoch 14/50, Train Loss: 2.0276, Train Accuracy: 28.94%
# Epoch 14/50, Test Accuracy: 26.80%
# Epoch 15/50, Train Loss: 2.0080, Train Accuracy: 30.01%
# Epoch 15/50, Test Accuracy: 26.99%

# lr = 0.000001
# Epoch 1/50, Train Loss: 2.2460, Train Accuracy: 16.39%
# Epoch 1/50, Test Accuracy: 24.66%
# Epoch 2/50, Train Loss: 2.1489, Train Accuracy: 23.83%
# Epoch 2/50, Test Accuracy: 26.41%
# Epoch 3/50, Train Loss: 2.1157, Train Accuracy: 24.56%
# Epoch 3/50, Test Accuracy: 25.24%
# Epoch 4/50, Train Loss: 2.0814, Train Accuracy: 26.75%
# Epoch 4/50, Test Accuracy: 30.68%
# Epoch 5/50, Train Loss: 1.9771, Train Accuracy: 30.30%
# Epoch 5/50, Test Accuracy: 31.84%
# Epoch 6/50, Train Loss: 1.8793, Train Accuracy: 33.80%
# Epoch 6/50, Test Accuracy: 33.59%
# Epoch 7/50, Train Loss: 1.8000, Train Accuracy: 35.51%
# Epoch 7/50, Test Accuracy: 34.95%
# Epoch 8/50, Train Loss: 1.7286, Train Accuracy: 38.62%
# Epoch 8/50, Test Accuracy: 34.56%
# Epoch 9/50, Train Loss: 1.7012, Train Accuracy: 39.49%
# Epoch 9/50, Test Accuracy: 40.39%
# Epoch 10/50, Train Loss: 1.5988, Train Accuracy: 43.58%
# Epoch 10/50, Test Accuracy: 40.97%
# Epoch 11/50, Train Loss: 1.5585, Train Accuracy: 44.36%
# Epoch 11/50, Test Accuracy: 44.47%
# Epoch 12/50, Train Loss: 1.5223, Train Accuracy: 46.16%
# Epoch 12/50, Test Accuracy: 42.14%
# Epoch 13/50, Train Loss: 1.4169, Train Accuracy: 51.95%
# Epoch 13/50, Test Accuracy: 47.38%
# Epoch 14/50, Train Loss: 1.3761, Train Accuracy: 50.54%
# Epoch 14/50, Test Accuracy: 49.32%
# Epoch 15/50, Train Loss: 1.3533, Train Accuracy: 51.61%
# Epoch 15/50, Test Accuracy: 51.26%
# Epoch 16/50, Train Loss: 1.2627, Train Accuracy: 56.03%
# Epoch 16/50, Test Accuracy: 51.26%
# Epoch 17/50, Train Loss: 1.1800, Train Accuracy: 59.78%
# Epoch 17/50, Test Accuracy: 51.26%
# Epoch 18/50, Train Loss: 1.0876, Train Accuracy: 63.57%
# Epoch 18/50, Test Accuracy: 50.87%
# Epoch 19/50, Train Loss: 1.1142, Train Accuracy: 60.80%
# Epoch 19/50, Test Accuracy: 57.09%
# Epoch 20/50, Train Loss: 0.9700, Train Accuracy: 68.43%
# Epoch 20/50, Test Accuracy: 52.43%
# Epoch 21/50, Train Loss: 0.9341, Train Accuracy: 67.95%
# Epoch 21/50, Test Accuracy: 58.64%
# Epoch 22/50, Train Loss: 0.8766, Train Accuracy: 71.35%
# Epoch 22/50, Test Accuracy: 54.17%
# Epoch 23/50, Train Loss: 0.8807, Train Accuracy: 71.01%
# Epoch 23/50, Test Accuracy: 59.22%
# Epoch 24/50, Train Loss: 0.7836, Train Accuracy: 74.37%
# Epoch 24/50, Test Accuracy: 64.27%
# Epoch 25/50, Train Loss: 0.7159, Train Accuracy: 76.56%
# Epoch 25/50, Test Accuracy: 61.94%
# Epoch 26/50, Train Loss: 0.6950, Train Accuracy: 77.29%
# Epoch 26/50, Test Accuracy: 62.14%
# Epoch 27/50, Train Loss: 0.6679, Train Accuracy: 77.33%
# Epoch 27/50, Test Accuracy: 62.14%
# Epoch 28/50, Train Loss: 0.5651, Train Accuracy: 82.15%
# Epoch 28/50, Test Accuracy: 62.14%
# Epoch 29/50, Train Loss: 0.5153, Train Accuracy: 83.71%
# Epoch 29/50, Test Accuracy: 63.50%
# Epoch 30/50, Train Loss: 0.5461, Train Accuracy: 83.71%
# Epoch 30/50, Test Accuracy: 56.31%
# Epoch 31/50, Train Loss: 0.5164, Train Accuracy: 83.61%
# Epoch 31/50, Test Accuracy: 60.39%
# Epoch 32/50, Train Loss: 0.4966, Train Accuracy: 84.63%
# Epoch 32/50, Test Accuracy: 62.14%
# Epoch 33/50, Train Loss: 0.3781, Train Accuracy: 89.11%
# Epoch 33/50, Test Accuracy: 64.47%
# Epoch 34/50, Train Loss: 0.4069, Train Accuracy: 87.16%
# Epoch 34/50, Test Accuracy: 61.36%
# Epoch 35/50, Train Loss: 0.3799, Train Accuracy: 88.08%
# Epoch 35/50, Test Accuracy: 63.50%
# Epoch 36/50, Train Loss: 0.3760, Train Accuracy: 88.57%
# Epoch 36/50, Test Accuracy: 64.47%
# Epoch 37/50, Train Loss: 0.2703, Train Accuracy: 93.34%
# Epoch 37/50, Test Accuracy: 65.24%
# Epoch 38/50, Train Loss: 0.2303, Train Accuracy: 94.02%
# Epoch 38/50, Test Accuracy: 65.44%
# Epoch 39/50, Train Loss: 0.2814, Train Accuracy: 91.39%
# Epoch 39/50, Test Accuracy: 64.27%
# Epoch 40/50, Train Loss: 0.2353, Train Accuracy: 92.95%
# Epoch 40/50, Test Accuracy: 67.96%
# Epoch 41/50, Train Loss: 0.2429, Train Accuracy: 92.66%
# Epoch 41/50, Test Accuracy: 63.88%
# Epoch 42/50, Train Loss: 0.2307, Train Accuracy: 93.53%
# Epoch 42/50, Test Accuracy: 66.99%
# Epoch 43/50, Train Loss: 0.1834, Train Accuracy: 95.53%
# Epoch 43/50, Test Accuracy: 65.63%
# Epoch 44/50, Train Loss: 0.1872, Train Accuracy: 95.33%
# Epoch 44/50, Test Accuracy: 67.38%
# Epoch 45/50, Train Loss: 0.2137, Train Accuracy: 93.73%
# Epoch 45/50, Test Accuracy: 65.44%
# Epoch 46/50, Train Loss: 0.1714, Train Accuracy: 95.14%
# Epoch 46/50, Test Accuracy: 62.91%
# Epoch 47/50, Train Loss: 0.2164, Train Accuracy: 93.34%
# Epoch 47/50, Test Accuracy: 65.44%
# Epoch 48/50, Train Loss: 0.1610, Train Accuracy: 95.43%
# Epoch 48/50, Test Accuracy: 66.02%
# Epoch 49/50, Train Loss: 0.1451, Train Accuracy: 96.45%
# Epoch 49/50, Test Accuracy: 65.63%
# Epoch 50/50, Train Loss: 0.1357, Train Accuracy: 96.25%
# Epoch 50/50, Test Accuracy: 60.58%
# Total training time: 2182.54 seconds
#
# Process finished with exit code 0

