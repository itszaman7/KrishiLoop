from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import base64
from .produce_price_predictor import ProducePricePredictor

class OrangeDetector:
    def __init__(self):
        self.fresh_model = None
        self.bad_model = None
        self.price_predictor = ProducePricePredictor()
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
        """Run detection with tier classification and price prediction"""
        print(f"\n=== Running Detection ===")
        print(f"Model: {'Bad Orange' if label_rename else 'Fresh Orange'}")
        print(f"Confidence Threshold: {conf_threshold}")
        
        if model is None:
            raise Exception("Model not loaded.")

        image = image.convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, conf=conf_threshold)
        detections = []

        print(f"\nDetections found: {len(results[0].boxes)}")
        
        # Plot results (will be in BGR)
        annotated_frame = results[0].plot(labels=False)  # Disable default labels
        # Convert back to RGB for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                label = int(box.cls[0])

                if label_rename and label == 1:
                    label = 0

                # Use price predictor for tier, price, and expiry
                tier = self.price_predictor.classify_tier(conf)
                predicted_price = self.price_predictor.predict_price(conf)
                expiry_date = self.price_predictor.predict_expiry(conf)
                market_desc = self.price_predictor.get_tier_description(tier)

                # Get coordinates for annotation
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.boxes[0][:4].tolist())

                # Calculate box dimensions
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Calculate adaptive font size (min: 0.4, max: 1.0)
                font_scale = min(max(min(box_width, box_height) / 300, 0.4), 1.0)
                thickness = max(int(font_scale * 2), 1)

                # Define colors and backgrounds based on tier
                tier_colors = {
                    'S': ((0, 100, 0), (144, 238, 144)),    # Dark Green text on Light Green bg
                    'A': ((0, 0, 139), (135, 206, 235)),    # Dark Blue text on Sky Blue bg
                    'B': ((139, 69, 19), (255, 218, 185)),  # Saddle Brown text on Peach bg
                    'C': ((139, 0, 0), (255, 192, 203)),    # Dark Red text on Pink bg
                    'R': ((69, 0, 69), (216, 191, 216))     # Dark Purple text on Light Purple bg
                }
                
                text_color, bg_color = tier_colors.get(tier, ((0, 0, 0), (200, 200, 200)))
                
                # Add custom label with tier
                label_text = f"{model.names[label] if not label_rename else 'orange_bad'} - Tier {tier}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )
                
                # Ensure text fits within image bounds
                padding = 5
                text_x1 = max(x1, padding)
                text_y1 = max(y1 - text_height - baseline - padding, padding)
                text_x2 = min(x1 + text_width + padding, annotated_frame.shape[1] - padding)
                
                # Draw background rectangle
                cv2.rectangle(
                    annotated_frame,
                    (text_x1 - padding, text_y1 - padding),
                    (text_x2, y1),
                    bg_color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (text_x1, y1 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness
                )

                detections.append({
                    'label': model.names[label] if not label_rename else "orange_bad",
                    'confidence': conf,
                    'coordinates': [x1, y1, x2, y2],
                    'tier': tier,
                    'predicted_price': round(predicted_price, 2),
                    'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                    'market_recommendation': market_desc
                })
        
        return detections, annotated_frame

    def process_image(self, image_file, conf_threshold=0.25):
        """Process image through both models and return results"""
        try:
            print("\n====== Starting Image Processing ======")
            print(f"Confidence Threshold: {conf_threshold}")
            
            image = Image.open(image_file)
            print("Image loaded successfully")
            
            print("\n=== Processing Fresh Oranges ===")
            fresh_detections, fresh_img = self.run_detection(
                image, 
                self.fresh_model, 
                conf_threshold
            )
            print(f"Fresh detections found: {len(fresh_detections)}")
            
            print("\n=== Processing Bad Oranges ===")
            bad_detections, bad_img = self.run_detection(
                image, 
                self.bad_model, 
                conf_threshold, 
                label_rename=True
            )
            print(f"Bad detections found: {len(bad_detections)}")
            
            # Get comprehensive analysis
            print("\n=== Analyzing Fresh Detections ===")
            fresh_analysis = self.price_predictor.analyze_detections(fresh_detections)
            
            print("\n=== Analyzing Bad Detections ===")
            bad_analysis = self.price_predictor.analyze_detections(bad_detections)
            
            print("\n====== Image Processing Complete ======")
            
            return {
                'success': True,
                'fresh_image': self._numpy_to_base64(fresh_img),
                'bad_image': self._numpy_to_base64(bad_img),
                'fresh_detections': fresh_detections,
                'bad_detections': bad_detections,
                'fresh_analysis': fresh_analysis,
                'bad_analysis': bad_analysis
            }
            
        except Exception as e:
            print(f"\nERROR in process_image: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _numpy_to_base64(self, img):
        """Convert numpy array to base64 string"""
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"