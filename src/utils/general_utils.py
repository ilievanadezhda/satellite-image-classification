import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_img_sizes(class_path):
    """ 
    Get the unique sizes of images in a class folder.
    """
    img_sizes = set()
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path)
            width, height = img.size
            img_sizes.add((width, height))
        except:
            print(f"Error opening {img_path}")
            continue
    return list(img_sizes)


def visualize_samples(class_path, title, num_samples=5):
    """ 
    Visualize a few samples from a class folder.
    """
    sample_images = os.listdir(class_path)[:num_samples]
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].axis('off')
        # axs[i].set_title(img_name)
    plt.tight_layout()
    plt.suptitle(f"Class: {title}")
    plt.show()


def extract_features(image_path):
    """ 
    Extract mean and standard deviation of RGB channels from an image file.
    """
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_array = np.array(image)
    
    # calculate the mean and standard deviation 
    mean_color = image_array.mean(axis=(0, 1))
    std_color = image_array.std(axis=(0, 1))
    
    return np.concatenate([mean_color, std_color])