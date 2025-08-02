import cv2
import face_recognition
from PIL import Image
import numpy as np
import os

class VisaPhotoProcessor:
    def __init__(self):
        # Constants
        self.DPI = 300
        self.PHOTO_WIDTH_MM = 33
        self.PHOTO_HEIGHT_MM = 48
        self.DESIRED_FACE_HEIGHT_MM = 30  # 28-33 mm middle value
        self.DESIRED_FACE_TOP_MARGIN_MM = 4
        self.DESIRED_FACE_BOTTOM_MARGIN_MM = 10
        
        # Calculate pixel dimensions
        self.PHOTO_WIDTH_PX = self._mm_to_pixels(self.PHOTO_WIDTH_MM)
        self.PHOTO_HEIGHT_PX = self._mm_to_pixels(self.PHOTO_HEIGHT_MM)
        self.DESIRED_FACE_HEIGHT_PX = self._mm_to_pixels(self.DESIRED_FACE_HEIGHT_MM)
        self.DESIRED_FACE_TOP_MARGIN_PX = self._mm_to_pixels(self.DESIRED_FACE_TOP_MARGIN_MM)
        self.DESIRED_FACE_BOTTOM_MARGIN_PX = self._mm_to_pixels(self.DESIRED_FACE_BOTTOM_MARGIN_MM)

    def _mm_to_inches(self, mm):
        """Convert millimeters to inches"""
        return mm / 25.4

    def _mm_to_pixels(self, mm):
        """Convert millimeters to pixels based on DPI"""
        return int(self._mm_to_inches(mm) * self.DPI)

    def detect_face(self, image_path):
        """Detect face in the image and return face information"""
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            raise ValueError("No face detected. Please use a clear frontal photo.")
            
        top, right, bottom, left = face_locations[0]
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
        face_width = right - left
        face_height = bottom - top
        
        return {
            'image': image,
            'face_location': face_locations[0],
            'center_x': face_center_x,
            'center_y': face_center_y,
            'width': face_width,
            'height': face_height
        }

    def remove_background(self, image_array, face_location):
        """Remove background using grabCut algorithm"""
        top, right, bottom, left = face_location
        
        # Initialize mask and models
        mask = np.zeros(image_array.shape[:2], np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        
        # Apply grabCut
        rect = (left, top, right - left, bottom - top)
        cv2.grabCut(image_array, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask and apply to image
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        result = image_array * mask2[:, :, np.newaxis]
        
        # Replace background with white
        white_bg = np.ones_like(image_array, dtype=np.uint8) * 255
        final_image = np.where(mask2[:, :, None] == 1, result, white_bg)
        
        return final_image

    def crop_and_resize(self, image, face_info):
        """Crop and resize image according to visa photo requirements"""
        # Calculate scaling factor based on desired face height
        scale = self.DESIRED_FACE_HEIGHT_PX / face_info['height']
        
        # Resize image
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Calculate new face center after resizing
        new_center_x = int(face_info['center_x'] * scale)
        new_center_y = int(face_info['center_y'] * scale)
        
        # Calculate desired face center position
        desired_face_center_y = self.DESIRED_FACE_TOP_MARGIN_PX + self.DESIRED_FACE_HEIGHT_PX // 2
        
        # Calculate crop coordinates
        crop_top = new_center_y - desired_face_center_y
        crop_left = new_center_x - self.PHOTO_WIDTH_PX // 2
        crop_bottom = crop_top + self.PHOTO_HEIGHT_PX
        crop_right = crop_left + self.PHOTO_WIDTH_PX
        
        # Add padding to handle edge cases
        padding = 300
        image_padded = cv2.copyMakeBorder(
            image_resized,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        # Perform cropping
        cropped = image_padded[
            crop_top + padding:crop_bottom + padding,
            crop_left + padding:crop_right + padding
        ]
        
        return cropped

    def save_image(self, image_array, output_path):
        """Save final image with proper DPI settings"""
        if image_array.shape[0] != self.PHOTO_HEIGHT_PX or image_array.shape[1] != self.PHOTO_WIDTH_PX:
            raise ValueError(f"Image dimensions must be {self.PHOTO_WIDTH_PX}x{self.PHOTO_HEIGHT_PX} pixels")
            
        image_pil = Image.fromarray(image_array)
        image_pil.save(output_path, dpi=(self.DPI, self.DPI))
        print(f"✅ Visa photo saved: {output_path}")

    def process_photo(self, input_path, output_path="china_visa_photo.jpg"):
        """Process photo according to Chinese visa requirements"""
        try:
            # Step 1: Detect face
            face_info = self.detect_face(input_path)
            
            # Step 2: Remove background
            image_no_bg = self.remove_background(face_info['image'], face_info['face_location'])
            
            # Step 3: Crop and resize
            final_image = self.crop_and_resize(image_no_bg, face_info)
            
            # Step 4: Save result
            self.save_image(final_image, output_path)
            
            return True
        except Exception as e:
            print(f"❌ Error processing photo: {str(e)}")
            return False


if __name__ == "__main__":
    processor = VisaPhotoProcessor()
    
    # Example usage
    input_path = "src/china_visa/EOSR9414.jpg"
    output_path = "china_visa_photo.jpg"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
    else:
        success = processor.process_photo(input_path, output_path)
        if success:
            print("Processing completed successfully")