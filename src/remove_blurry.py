import cv2
import os
import numpy as np


def is_blurry(image_path, threshold=50.0):
    """
    Determines if an image is blurry based on the Laplacian variance method.
    
    Parameters:
    - image_path (str): Path to the image file.
    - threshold (float): The variance threshold below which the image is considered blurry.
    
    Returns:
    - bool: True if the image is blurry, False otherwise.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Error reading the image {image_path}")
    
    # Compute the Laplacian of the image and then compute the variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    
    # If the variance is below the threshold, the image is blurry
    return variance < threshold


def remove_blurry_images(directory, threshold=60.0):
    """
    Removes blurry images from a directory.
    
    Parameters:
    - directory (str): Path to the directory containing images.
    - threshold (float): The variance threshold below which the image is considered blurry.
    """
    # Get all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if the file is an image (you can extend this list to include more image formats)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                if is_blurry(file_path, threshold):
                    print(f"Removing blurry image: {filename}")
                    os.remove(file_path)
            except ValueError as e:
                print(e)

# Example usage
image_directory = '/users/tigerhu/Documents/test_photos/'
remove_blurry_images(image_directory)
