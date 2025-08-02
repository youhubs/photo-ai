import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Define the processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the directories
image_dir = "/Users/tigerhu/projects/photo-ai/photos"  # Source directory
good_dir = "/Users/tigerhu/projects/photo-ai/photo-good"  # Destination for good photos
bad_dir = "/Users/tigerhu/projects/photo-ai/photo-bad"  # Destination for bad photos

# Create the directories if they don't exist
os.makedirs(good_dir, exist_ok=True)
os.makedirs(bad_dir, exist_ok=True)

try:
    # Get list of image file paths
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Debugging: Confirm the files
    print("Image files found:", image_files)
    if not image_files:
        raise ValueError(f"No valid image files found in {image_dir}")

    # Load the images into PIL Image objects
    images = []
    for file in image_files:
        try:
            img = Image.open(file).convert("RGB")
            images.append(img)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not images:
        raise ValueError("No images could be loaded successfully")

    # Process text and images
    inputs = processor(
        text=["a well-composed photo", "a poorly composed photo"],
        images=images,
        return_tensors="pt",  # PyTorch tensors
        padding=True
    )

    # Run the model
    outputs = model(**inputs)

    # Extract logits and convert to probabilities
    logits_per_image = outputs.logits_per_image  # Shape: [num_images, num_texts]
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Analyze and save photos to appropriate folders
    good_count = 0
    bad_count = 0
    for i, file in enumerate(image_files):
        well_composed_score = probs[i][0].item()  # Probability for "well-composed"
        poorly_composed_score = probs[i][1].item()  # Probability for "poorly composed"
        print(f"\n{file}:")
        print(f"  Well-composed: {well_composed_score:.4f}")
        print(f"  Poorly composed: {poorly_composed_score:.4f}")

        # Decide where to save the photo
        if well_composed_score > poorly_composed_score:
            print("  -> Classified as a well-composed photo")
            dest_path = os.path.join(good_dir, os.path.basename(file))
            shutil.copy2(file, dest_path)  # Copy to photo-good
            print(f"  -> Saved to: {dest_path}")
            good_count += 1
        else:
            print("  -> Classified as a poorly composed photo")
            dest_path = os.path.join(bad_dir, os.path.basename(file))
            shutil.copy2(file, dest_path)  # Copy to photo-bad
            print(f"  -> Saved to: {dest_path}")
            bad_count += 1

    print(f"\nTotal well-composed photos saved: {good_count}")
    print(f"Total poorly composed photos saved: {bad_count}")

except Exception as e:
    print(f"An error occurred: {e}")