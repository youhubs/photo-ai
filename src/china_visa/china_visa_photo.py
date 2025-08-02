import cv2
import face_recognition
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt  # For optional visualization

class VisaPhotoProcessor:
    def __init__(self):
        # Constants
        self.DPI = 300
        self.PHOTO_WIDTH_MM = 33
        self.PHOTO_HEIGHT_MM = 48
        self.DESIRED_FACE_HEIGHT_MM = 30  # 28-33 mm middle value
            # New flexible ranges (adjust these as needed)
        self.MIN_FACE_HEIGHT_MM = 19  # Reduced from 28
        self.MAX_FACE_HEIGHT_MM = 36  # Increased from 33
        self.MIN_FACE_PERCENT = 0.25  # Face must occupy at least 25% of image height
    
        self.DESIRED_FACE_TOP_MARGIN_MM = 4
        self.DESIRED_FACE_BOTTOM_MARGIN_MM = 10
        
        # Calculate pixel dimensions
        self.PHOTO_WIDTH_PX = self._mm_to_pixels(self.PHOTO_WIDTH_MM)
        self.PHOTO_HEIGHT_PX = self._mm_to_pixels(self.PHOTO_HEIGHT_MM)
        self.DESIRED_FACE_HEIGHT_PX = self._mm_to_pixels(self.DESIRED_FACE_HEIGHT_MM)
        self.DESIRED_FACE_TOP_MARGIN_PX = self._mm_to_pixels(self.DESIRED_FACE_TOP_MARGIN_MM)
        self.DESIRED_FACE_BOTTOM_MARGIN_PX = self._mm_to_pixels(self.DESIRED_FACE_BOTTOM_MARGIN_MM)

    def _mm_to_inches(self, mm):
        return mm / 25.4

    def _mm_to_pixels(self, mm):
        return int(self._mm_to_inches(mm) * self.DPI)

    def _pixels_to_mm(self, pixels):
        return pixels / self.DPI * 25.4

    def detect_face(self, image_path):
        """Complete face detection with all required fields"""
        try:
            # Load image and detect faces
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                raise ValueError("No face detected. Please use a clear frontal photo")
            
            # Get largest face
            top, right, bottom, left = max(face_locations,
                                        key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
            
            # Calculate all required measurements
            face_height = bottom - top
            face_width = right - left
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            # Add safety margin (25% of face size)
            margin = int(max(face_height, face_width) * 0.25)
            top = max(0, top - margin)
            bottom = min(image.shape[0], bottom + margin)
            left = max(0, left - margin)
            right = min(image.shape[1], right + margin)
            
            # Return complete face info
            return {
                'original_image': image,
                'face_box': (top, right, bottom, left),
                'face_height': bottom - top,
                'face_width': right - left,
                'center_x': (left + right) // 2,
                'center_y': (top + bottom) // 2,
                'original_height': image.shape[0],
                'original_width': image.shape[1]
            }
            
        except Exception as e:
            raise ValueError(f"Face detection failed: {str(e)}")


    def validate_face_info(self, face_info):
        """Ensure all required fields are present"""
        required_fields = [
            'original_image', 'face_box', 'face_height',
            'face_width', 'center_x', 'center_y',
            'original_height', 'original_width'
        ]
        missing = [field for field in required_fields if field not in face_info]
        if missing:
            raise ValueError(f"Missing face info fields: {', '.join(missing)}")
        return True

    def crop_to_requirements(self, image, face_info):
        """Safe cropping with complete validation"""
        try:
            # First validate face info structure
            self.validate_face_info(face_info)
            
            # Calculate scaling factor
            scale = self.DESIRED_FACE_HEIGHT_PX / face_info['face_height']
            
            # Resize image
            resized = cv2.resize(image, 
                            (int(image.shape[1] * scale), 
                                int(image.shape[0] * scale)))
            
            # Calculate new face center
            new_center_x = int(face_info['center_x'] * scale)
            new_center_y = int(face_info['center_y'] * scale)
            
            # Calculate crop area
            crop_top = new_center_y - self.DESIRED_FACE_TOP_MARGIN_PX - self.DESIRED_FACE_HEIGHT_PX//2
            crop_left = new_center_x - self.PHOTO_WIDTH_PX//2
            crop_bottom = crop_top + self.PHOTO_HEIGHT_PX
            crop_right = crop_left + self.PHOTO_WIDTH_PX
            
            # Add padding to handle edge cases
            padding = max(self.PHOTO_HEIGHT_PX, self.PHOTO_WIDTH_PX)
            padded = cv2.copyMakeBorder(resized,
                                    padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255])
            
            # Perform cropping
            cropped = padded[
                crop_top+padding : crop_bottom+padding,
                crop_left+padding : crop_right+padding
            ]
            
            # Final size validation
            if cropped.shape[0] != self.PHOTO_HEIGHT_PX or cropped.shape[1] != self.PHOTO_WIDTH_PX:
                cropped = cv2.resize(cropped, (self.PHOTO_WIDTH_PX, self.PHOTO_HEIGHT_PX))
            
            return cropped
            
        except Exception as e:
            raise ValueError(f"Cropping failed: {str(e)}")
        
    def remove_background(self, image_array):
        """Optimized background removal for cropped images"""
        try:
            # Convert to OpenCV format
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            h, w = image.shape[:2]
            
            # Create mask with face in center
            mask = np.zeros((h, w), np.uint8)
            
            # Define face area (center 50-70% of image)
            face_size = int(min(h, w) * 0.6)
            center_x, center_y = w//2, h//2
            top = max(0, center_y - face_size//2)
            bottom = min(h, center_y + face_size//2)
            left = max(0, center_x - face_size//2)
            right = min(w, center_x + face_size//2)
            
            # Set face area
            mask[top:bottom, left:right] = cv2.GC_PR_FGD
            
            # Set inner area as definite foreground
            inner_size = int(face_size * 0.7)
            inner_top = max(0, center_y - inner_size//2)
            inner_bottom = min(h, center_y + inner_size//2)
            inner_left = max(0, center_x - inner_size//2)
            inner_right = min(w, center_x + inner_size//2)
            mask[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_FGD
            
            # Apply grabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            
            # Apply mask
            foreground = cv2.bitwise_and(image, image, mask=final_mask)
            white_bg = np.ones_like(image, dtype=np.uint8) * 255
            result = cv2.bitwise_or(foreground, white_bg, mask=cv2.bitwise_not(final_mask))
            result = cv2.add(foreground, result)
            
            # Convert back to RGB
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Background removal failed: {str(e)}")

    def validate_result(self, image_array):
        """Comprehensive output validation"""
        try:
            # 1. Check dimensions
            if image_array.shape[0] != self.PHOTO_HEIGHT_PX or image_array.shape[1] != self.PHOTO_WIDTH_PX:
                raise ValueError(f"Invalid dimensions. Expected {self.PHOTO_WIDTH_PX}x{self.PHOTO_HEIGHT_PX}, got {image_array.shape[1]}x{image_array.shape[0]}")
            
            # 2. Check background is white
            corners = [
                image_array[0,0], image_array[0,-1],
                image_array[-1,0], image_array[-1,-1]
            ]
            for corner in corners:
                if not np.all(corner == [255,255,255]):
                    raise ValueError("Background not pure white in all corners")
            
            # 3. Check face presence in center
            center_x = self.PHOTO_WIDTH_PX // 2
            center_y = self.PHOTO_HEIGHT_PX // 2
            center_region = image_array[center_y-50:center_y+50, center_x-50:center_x+50]
            if np.mean(center_region) > 240:  # Too white
                raise ValueError("No face detected in center region")
            
            # 4. Check face proportions
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("Cannot detect face contours")
                
            largest_contour = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest_contour)
            
            face_height_mm = self._pixels_to_mm(h)
            if not (19 <= face_height_mm <= 33):
                raise ValueError(f"Face height {face_height_mm:.1f}mm outside required 28-33mm range")
            
            return True
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")

    def generate_debug_visualizations(self, original, processed, face_info, output_dir="debug"):
        """Create visual verification files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Mark face on original image
        marked_original = original.copy()
        top, right, bottom, left = face_info['face_box']
        cv2.rectangle(marked_original, (left, top), (right, bottom), (255,0,0), 2)
        
        # 2. Show crop area
        crop_preview = original.copy()
        scale = self.DESIRED_FACE_HEIGHT_PX / face_info['face_height']
        new_center_x = int(face_info['center_x'] * scale)
        new_center_y = int(face_info['center_y'] * scale)
        crop_left = new_center_x - self.PHOTO_WIDTH_PX // 2
        crop_top = new_center_y - (self.DESIRED_FACE_TOP_MARGIN_PX + self.DESIRED_FACE_HEIGHT_PX // 2)
        
        # Draw on original scale
        cv2.rectangle(crop_preview, 
                    (int(crop_left/scale), int(crop_top/scale)),
                    (int((crop_left+self.PHOTO_WIDTH_PX)/scale), int((crop_top+self.PHOTO_HEIGHT_PX)/scale)),
                    (0,0,255), 2)
        
        # 3. Save debug images
        cv2.imwrite(f"{output_dir}/1_original_with_face.jpg", cv2.cvtColor(marked_original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/2_crop_area_preview.jpg", cv2.cvtColor(crop_preview, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/3_final_result.jpg", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        
        # 4. Create report
        report = f"""Visa Photo Processing Report
Dimensions: {self.PHOTO_WIDTH_PX}x{self.PHOTO_HEIGHT_PX} pixels ({self.PHOTO_WIDTH_MM}x{self.PHOTO_HEIGHT_MM}mm)
Face Height: {self._pixels_to_mm(face_info['face_height']):.1f}mm
Face Position: Top={top}, Bottom={bottom}, Left={left}, Right={right}
Validation: {"PASSED" if self.validate_result(processed) else "FAILED"}"""
        
        with open(f"{output_dir}/processing_report.txt", "w") as f:
            f.write(report)

    def save_image(self, image_array, output_path):
        """Save with validation and metadata"""
        try:
            self.validate_result(image_array)
            image_pil = Image.fromarray(image_array)
            image_pil.save(output_path, dpi=(self.DPI, self.DPI), quality=100)
            print(f"✅ Saved visa photo to {output_path}")
        except Exception as e:
            raise ValueError(f"Failed to save image: {str(e)}")

    def process_photo(self, input_path, output_path="visa_photo.jpg", debug=False):
        """Enhanced processing with auto-resizing"""
        try:
            print(f"Processing {input_path}...")
            
            # 1. Load image with size validation
            print("- Validating image size...")
            image = face_recognition.load_image_file(input_path)
            if min(image.shape[0], image.shape[1]) < 1000:
                raise ValueError(
                    f"Image too small ({image.shape[1]}x{image.shape[0]}). "
                    "Minimum 1200x1600 pixels required."
                )
            
            # 2. Face detection with retry logic
            print("- Detecting face...")
            face_info = None
            for attempt in [1.0, 1.3, 1.6]:  # Try different scales
                try:
                    resized = cv2.resize(image, (0,0), fx=attempt, fy=attempt)
                    face_locations = face_recognition.face_locations(resized)
                    
                    if face_locations:
                        top, right, bottom, left = max(face_locations, 
                            key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
                        face_height = bottom - top
                        if self._pixels_to_mm(face_height/attempt) >= 15:
                            face_info = {
                                'original_image': image,
                                'face_box': (
                                    int(top/attempt), int(right/attempt),
                                    int(bottom/attempt), int(left/attempt)
                                ),
                                'face_height': face_height/attempt,
                                'original_height': image.shape[0],
                                'original_width': image.shape[1]
                            }
                            break
                except:
                    continue
                    
            if not face_info:
                raise ValueError(
                    "Face too small in photo. Please:\n"
                    "1. Move closer (face should fill 30% of photo)\n"
                    "2. Use higher resolution camera\n"
                    "3. Ensure good lighting on face"
                )
            
            # Rest of processing pipeline...
            print("- Cropping to visa dimensions...")
            cropped = self.crop_to_requirements(face_info['original_image'], face_info)
            
            print("- Removing background...")
            final = self.remove_background(cropped)
            
            print("- Validating result...")
            self.validate_result(final)
            self.save_image(final, output_path)
            
            if debug:
                self.generate_debug_visualizations(image, final, face_info)
            
            print("✅ Success! Visa photo created at:", os.path.abspath(output_path))
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if debug and 'face_box' in locals():
                self.generate_debug_visualizations(image, None, face_info)
            return False


# Example Usage
if __name__ == "__main__":
    processor = VisaPhotoProcessor()

    try:
        # 1. Detect face with all required fields
        face_info = processor.detect_face("hxy.jpg")
        
        # 2. Process the image
        cropped = processor.crop_to_requirements(face_info['original_image'], face_info)
        final = processor.remove_background(cropped)
        processor.save_image(final, "visa_photo.jpg")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Please check:")
        print("- Photo shows clear frontal face")
        print("- Face takes up 30-40% of photo height")
        print("- Image resolution is at least 1200x1600 pixels")