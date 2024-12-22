from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import base64

class OrangeDetector:
    def __init__(self):
        self.fresh_model = None
        self.bad_model = None
        self.load_models()

    def load_models(self):
        """Load YOLO models for fresh and bad orange detection"""
        try:
            self.fresh_model = YOLO("models/good/fresh_oranges.pt")
            self.bad_model = YOLO("models/bad/bad_oranges.pt")
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def run_detection(self, image, model, conf_threshold, label_rename=None):
        """Run detection with label renaming option"""
        if model is None:
            raise Exception("Model not loaded.")

        # Convert PIL Image to BGR numpy array (what YOLO expects)
        image = image.convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, conf=conf_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                label = int(box.cls[0])

                if label_rename and label == 1:  # Relabel "saine" to "orange_bad"
                    label = 0

                detections.append({
                    'label': model.names[label] if not label_rename else "orange_bad",
                    'confidence': conf,
                    'coordinates': box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.boxes[0][:4].tolist()
                })

        # Plot results (will be in BGR)
        annotated_frame = results[0].plot()
        # Convert back to RGB for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return detections, annotated_frame

    def process_image(self, image_file, conf_threshold=0.25):
        """Process image through both models and return results"""
        try:
            # Open image
            image = Image.open(image_file)
            
            # Run detection on both models
            fresh_detections, fresh_img = self.run_detection(
                image, 
                self.fresh_model, 
                conf_threshold
            )
            
            bad_detections, bad_img = self.run_detection(
                image, 
                self.bad_model, 
                conf_threshold, 
                label_rename=True
            )
            
            # Convert images to base64
            fresh_image_b64 = self._numpy_to_base64(fresh_img)
            bad_image_b64 = self._numpy_to_base64(bad_img)
            
            return {
                'success': True,
                'fresh_image': fresh_image_b64,
                'bad_image': bad_image_b64,
                'fresh_detections': fresh_detections,
                'bad_detections': bad_detections
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _numpy_to_base64(self, img):
        """Convert numpy array to base64 string"""
        # Image is in RGB, convert to BGR for cv2.imencode
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}" 