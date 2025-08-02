import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Define the processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the image directory
image_dir = "/Users/tigerhu/projects/photo-ai/photos"  # Adjust this path as needed

try:
    # Get list of image file paths
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

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

    # Debugging: Check the processed inputs (optional)
    # print("Processed inputs:", inputs)

    # Run the model
    outputs = model(**inputs)

    # Extract logits (similarity scores between text and images)
    logits_per_image = outputs.logits_per_image  # Shape: [num_images, num_texts]
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Print results
    print("\nSimilarity Scores (logits):")
    print(logits_per_image)
    print("\nProbabilities:")
    print(probs)

    # Analyze results
    for i, file in enumerate(image_files):
        well_composed_score = probs[i][0].item()  # Probability for "well-composed"
        poorly_composed_score = probs[i][1].item()  # Probability for "poorly composed"
        print(f"\n{file}:")
        print(f"  Well-composed: {well_composed_score:.4f}")
        print(f"  Poorly composed: {poorly_composed_score:.4f}")
        if well_composed_score > poorly_composed_score:
            print("  -> Likely a well-composed photo")
        else:
            print("  -> Likely a poorly composed photo")

except Exception as e:
    print(f"An error occurred: {e}")
