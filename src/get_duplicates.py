import os
import piexif
from PIL import Image
from collections import defaultdict
from datetime import datetime, timedelta

def get_capture_time(image_path):
    """
    Extract the capture time from the EXIF data of an image.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - datetime: The capture time of the image.
    """
    try:
        # Open image and get EXIF data
        image = Image.open(image_path)
        exif_data = piexif.load(image.info['exif'])
        
        # Extract 'DateTimeOriginal' tag (0x9003)
        datetime_str = exif_data.get('0th', {}).get(piexif.ImageIFD.DateTime, None)
        if datetime_str is not None:
            # Convert to datetime object
            capture_time = datetime.strptime(datetime_str.decode(), '%Y:%m:%d %H:%M:%S')
            return capture_time
    except Exception as e:
        print(f"Error reading EXIF for {image_path}: {e}")
    
    return None

def group_images_by_time(directory, time_delta_seconds=2):
    """
    Groups images by capture time based on EXIF metadata (within a given time delta).
    
    Parameters:
    - directory (str): Path to the directory containing the images.
    - time_delta_seconds (int): Maximum time difference (in seconds) between images to be considered in the same group.
    
    Returns:
    - dict: A dictionary where keys are group identifiers and values are lists of image paths.
    """
    images_by_time = defaultdict(list)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            capture_time = get_capture_time(file_path)
            
            if capture_time is not None:
                # Create a key for grouping based on timestamp, rounded to nearest `time_delta_seconds`
                timestamp_key = capture_time.replace(second=(capture_time.second // time_delta_seconds) * time_delta_seconds)
                
                # Add image to the appropriate group based on timestamp
                images_by_time[timestamp_key].append(file_path)
    
    return images_by_time

def display_groups(groups):
    """
    Display the groups of images.
    
    Parameters:
    - groups (dict): The groups of images to display.
    """
    for timestamp, image_paths in groups.items():
        print(f"Group with timestamp {timestamp}:")
        for image_path in image_paths:
            print(f"  {image_path}")
        print()  # Blank line between groups

# Example usage
# image_directory = '/users/tigerhu/Documents/test_photos/'
image_directory = '/Volumes/Untitled/DCIM/186_1222/masterclasses'
groups = group_images_by_time(image_directory)

# Display groups
display_groups(groups)
