
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
import cv2
from skimage.transform import resize
import random
from torchvision import models, transforms
from torch import nn
# Define your transformation
transform = transforms.Compose([
    # Add your desired transformations here
    transforms.Resize((256, 256)),  # Example: Resize images to 256x256
    transforms.ToTensor(),  # Convert image to tensor
    # Add more transformations as needed
])

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



def skeleton_feature(image):

    random_number = random.random()
    image_array = np.array(image)
    skeleton_array = skeletonize(image_array)
    skeleton_array = cv2.dilate(skeleton_array, np.ones((3, 3), np.uint8))
    skeleton_array_56 = resize(skeleton_array, (56, 56), mode='constant', anti_aliasing=True)
    if random_number < 0.5:
        skeleton_feature =  skeleton_array_56.astype(np.float32)
    else:
        skeleton_feature = 1 - skeleton_array_56.astype(np.float32)

    return skeleton_feature

def convolution_feature(image):

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    # Ensure the image has three channels (RGB)
                                    transforms.Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(image)
    img = img.unsqueeze(0)

    model = models.resnet18(pretrained=True)  # 可以改为model_ft = models.resnet18(pretrained=True)，直接下载ResNet18.
    num_classes = 10  # Change this to match the number of output classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Optionally freeze the weights of the pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    new_model = nn.Sequential(*list(model.children())[:5])
    convolution_feature = new_model(img)
    return convolution_feature

# Original directory containing 10 subfolders
original_dir = 'D:\DataSet\JiDaTop10'

# New directory to save transformed images
new_dir = 'D:\DataSet\JiDaTop10_Con_Ske_Aug'

# Create the new directory if it doesn't exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Traverse each folder in the original directory
for folder_name in os.listdir(original_dir):
    original_folder = os.path.join(original_dir, folder_name)
    new_folder = os.path.join(new_dir, folder_name)

    # Create the corresponding folder in the new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Traverse each image in the original folder
    for filename in os.listdir(original_folder):
        original_image_path = os.path.join(original_folder, filename)
        new_image_path = os.path.join(new_folder, filename)

        # Open the image
        image = Image.open(original_image_path)

        f3 = convolution_feature(image)
        ske_feature = skeleton_feature(image)

        feature_maps = f3.permute((1, 0, 2, 3))
        f3_skele = torch.zeros_like(feature_maps)
        f3_skele = f3_skele.permute((1, 0, 2, 3))
        for i in range(feature_maps.size(0)):
            feature_map = feature_maps[i, 0]  # Assuming each feature map is a single-channel image
            # Detach the tensor from the computation graph and convert it to a NumPy array
            feature_map_numpy = feature_map.detach().numpy()
            composed_numpy = feature_map_numpy + ske_feature
            composed_numpy2 = map_pixels(ske_feature, composed_numpy)

            image_array_uint8 = (composed_numpy2* 255.).astype(np.uint8)

            new_image = Image.fromarray(image_array_uint8)
            new_image_name = str(i)+'.jpg'
            new_image_path = os.path.join(new_folder, new_image_name)

            new_image.save(new_image_path,format='JPEG')



print("Transformation and saving complete.")
