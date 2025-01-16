import tensorflow as tf
import numpy as np
import cv2

# Load pre-trained NIMA model (can be found on TensorFlow Hub or other sources)
model = tf.keras.models.load_model("path_to_your_model")

def predict_sharpness(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to the input size expected by the model
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    
    # Predict sharpness score using the model
    prediction = model.predict(image)
    
    # Score interpretation - You may want to adjust this based on the model output
    sharpness_score = prediction[0]  # This will depend on the model output format
    
    return sharpness_score

# Example usage
image_path = 'path_to_your_image.jpg'
sharpness_score = predict_sharpness(image_path)

if sharpness_score > 0.5:  # You can adjust this threshold based on the score range
    print(f"Image {image_path} is sharp.")
else:
    print(f"Image {image_path} is blurry.")
