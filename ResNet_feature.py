import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2
from skimage.transform import resize
import json


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

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def map_pixels(matrix_A, matrix_B):
    # Make a copy of matrix_B to keep the original intact
    mapped_matrix_B = np.copy(matrix_B)

    # Find coordinates of pixels with value 0 in matrix A
    zero_indices = np.argwhere(matrix_A == 0)

    # Define the input range for mapping
    # min_input = np.min(matrix_B)
    # max_input = np.max(matrix_B)

    min_input = 0
    max_input = 1

    # Define the output range for mapping
    min_output = 0
    max_output = 1

    # Perform linear transformation for each pixel
    for index in zero_indices:
        row, col = index
        input_value = matrix_B[row, col]
        output_value = min_output + (input_value - min_input) * (max_output - min_output) / (max_input - min_input)
        mapped_matrix_B[row, col] = output_value

    return mapped_matrix_B


#image = Image.open(r'D:\Paper\论文\我的论文\ICME\image\original.jpg')
# Load your image with error handling
try:
    image_grey = Image.open(r'D:\Paper\论文\我的论文\ICME\image\original.jpg').convert('L')  # Convert to grayscale
    image = Image.open(r'D:\Paper\论文\我的论文\ICME\image\original.jpg')
except FileNotFoundError:
    print("File not found.")
    exit()
except Exception as e:
    print("An error occurred while loading the image:", e)
    exit()

image_array = np.array(image_grey)

# Perform skeleton extraction
skeleton_array = skeletonize(image_array)
skeleton_array = cv2.dilate(skeleton_array, np.ones((3, 3), np.uint8))
skeleton_array_56 = resize(skeleton_array, (56,56), mode='constant', anti_aliasing=True)
skeleton_array_56 = skeleton_array_56.astype(np.float32)
skeleton_array_56_reverse = 1-skeleton_array_56.astype(np.float32)

#---------------------------------------------------------
model = models.resnet18(pretrained=True)#可以改为model_ft = models.resnet18(pretrained=True)，直接下载ResNet18.
print(model)
num_classes = 10  # Change this to match the number of output classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Optionally freeze the weights of the pretrained layers
for param in model.parameters():
    param.requires_grad = False
print(model)



transform = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                        # Ensure the image has three channels (RGB)
                         transforms.Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
img = transform(image)
img = img.unsqueeze(0)



new_model = nn.Sequential(*list(model.children())[:5])
f3 = new_model(img)
#save_img(f3, 'layer1')


new_model = nn.Sequential(*list(model.children())[:6])
f4 = new_model(img)  # [1, 128, 28, 28]
#save_img(f4, 'layer2')

new_model = nn.Sequential(*list(model.children())[:7])
print(new_model)
f5 = new_model(img)  # [1, 256, 14, 14]
print(f5.shape)
#save_img(f5, 'layer3')

new_model = nn.Sequential(*list(model.children())[:8])
print(new_model)
f6 = new_model(img)  # [1, 256, 14, 14]
print(f6.shape)
#save_img(f6, 'layer4')

feature_maps = f3.permute((1, 0, 2, 3))
f3_skele = torch.zeros_like(feature_maps)
f3_skele = f3_skele.permute((1, 0, 2, 3))
for i in range(feature_maps.size(0)):
    # Access the ith feature map
    feature_map = feature_maps[i, 0]  # Assuming each feature map is a single-channel image

    # Detach the tensor from the computation graph and convert it to a NumPy array
    feature_map_numpy = feature_map.detach().numpy()
    random_number = random.random()

    # Check if the random number is less than 0.5 (50%)
    if random_number < 0.5:
        composed_numpy = feature_map_numpy + skeleton_array_56_reverse


    else:
        composed_numpy = feature_map_numpy+skeleton_array_56
    composed_numpy1 = cv2.add(feature_map_numpy,skeleton_array_56)
    #composed_numpy2 = cv2.bitwise_and(feature_map_numpy,skeleton_array_56)
    composed_numpy2 = map_pixels(skeleton_array_56, composed_numpy)

    tensor_to_add = torch.tensor(composed_numpy2)
    f3_skele[:, i:i + 1, :, :] = tensor_to_add
print(f3_skele.size())
save_img(f3_skele, 'layer1_skele_map_down_00to10_Random_Rerverse')

    #
    # plt.figure(figsize=(10, 10))
    #
    # new_tensor = torch.empty(1, 1, 56, 56)
    #
    # combined_tensor = torch.cat((combined_tensor, tensor_to_add), dim=1)
    #
    # # Plot the first array on the left side
    # plt.subplot(2, 2, 1)
    # plt.imshow(feature_map_numpy, cmap='gray')
    # plt.title('feature_map_numpy')
    #
    # # Plot the second array on the right side
    # plt.subplot(2, 2, 2)
    # plt.imshow(composed_numpy, cmap='gray')
    # plt.title('composed_numpy')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(composed_numpy2, cmap='gray')
    # plt.title('composed_numpy2')
    #
    # # Plot the second array on the right side
    # plt.subplot(2, 2, 4)
    # plt.imshow(skeleton_array_56, cmap='gray')
    # plt.title('skeleton_array_56')
    #
    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    #
    # # Display the plot
    # plt.show()
    #
    #
    #
    # # Display the feature map
    # plt.imshow(feature_map_numpy, cmap='gray')
    # plt.title("Feature Map {}".format(i))
    # plt.show()