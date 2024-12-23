from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import base64
import os
import psutil
import platform
from datetime import datetime

class RealtimeDetector:
    def __init__(self):
        try:
            # Initialize variables
            self.fresh_model = None
            self.bad_model = None
            
            # Get system specs
            self.system_info = self.get_system_specs()
            
            # Automatically detect best device and optimize settings
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.conf_threshold = 0.25  # Increased slightly for better detection
                self.input_size = (640, 640)  # Back to original size
                self.scale_percent = 75  # Increased for better detection
                self.jpeg_quality = 85  # Increased quality
            else:
                self.device = 'cpu'
                self.conf_threshold = 0.35
                self.input_size = (416, 416)
                self.scale_percent = 50
                self.jpeg_quality = 75

            self.iou_threshold = 0.4
            self.max_det = 10  # Maximum detections per frame
            print(f"Detector initialized on {self.device}")
            
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            raise

    def get_system_specs(self):
        """Get system specifications"""
        try:
            cpu_freq = psutil.cpu_freq()
            specs = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'cpu_freq_max': round(cpu_freq.max, 2) if cpu_freq else "N/A",
                'ram_total': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                'settings': {
                    'device': self.device,
                    'input_size': self.input_size,
                    'scale_percent': self.scale_percent,
                    'conf_threshold': self.conf_threshold
                }
            }
            return specs
        except Exception as e:
            print(f"Error getting system specs: {str(e)}")
            return {}

    def get_performance_metrics(self):
        """Get current performance metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'ram_percent': psutil.virtual_memory().percent,
                'gpu_memory_used': torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        except Exception as e:
            print(f"Error getting performance metrics: {str(e)}")
            return {}

    def load_model(self, model_type):
        """Load model on demand with optimized settings"""
        try:
            if model_type == 'fresh' and self.fresh_model is None:
                self.fresh_model = YOLO("models/good/fresh_oranges.pt")
                self.fresh_model.to(self.device)
            elif model_type == 'bad' and self.bad_model is None:
                self.bad_model = YOLO("models/bad/bad_oranges.pt")
                self.bad_model.to(self.device)
            elif model_type == 'both':
                if self.fresh_model is None:
                    self.fresh_model = YOLO("models/good/fresh_oranges.pt")
                    self.fresh_model.to(self.device)
                if self.bad_model is None:
                    self.bad_model = YOLO("models/bad/bad_oranges.pt")
                    self.bad_model.to(self.device)
            print(f"Model {model_type} loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model {model_type}: {str(e)}")
            raise

    def process_frame(self, frame_data, model_type='both'):
        """Process a single frame with performance monitoring"""
        try:
            start_time = datetime.now()
            
            # Validate input
            if not isinstance(frame_data, str) or not frame_data.startswith('data:image'):
                print("Invalid frame data format")
                return {
                    'success': False,
                    'error': 'Invalid frame data format'
                }
            
            try:
                # Split the data URI and get the base64 part
                header, encoded = frame_data.split(',', 1)
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(encoded)
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Decode image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("Failed to decode image")
                
            except Exception as e:
                print(f"Error decoding frame: {str(e)}")
                return {
                    'success': False,
                    'error': f'Error decoding image: {str(e)}'
                }

            # Load models if needed
            self.load_model(model_type)
            
            # Process frame
            original_height, original_width = frame.shape[:2]
            
            # Resize while maintaining aspect ratio
            process_width = min(640, original_width)
            scale = process_width / original_width
            process_height = int(original_height * scale)
            
            frame_resized = cv2.resize(frame, (process_width, process_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            fresh_detections = []
            bad_detections = []

            # Run detections
            if model_type in ['fresh', 'both']:
                results = self.fresh_model(
                    frame_rgb,
                    conf=self.conf_threshold,
                    device=self.device,
                    max_det=self.max_det
                )
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        # Scale coordinates back to original size
                        scaled_coords = coords / scale
                        fresh_detections.append({
                            'coordinates': scaled_coords.tolist(),
                            'confidence': float(box.conf),
                            'label': 'orange_fresh'
                        })

            if model_type in ['bad', 'both']:
                results = self.bad_model(
                    frame_rgb,
                    conf=self.conf_threshold,
                    device=self.device,
                    max_det=self.max_det
                )
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        scaled_coords = coords / scale
                        bad_detections.append({
                            'coordinates': scaled_coords.tolist(),
                            'confidence': float(box.conf),
                            'label': 'orange_bad'
                        })

            # Process detections
            final_detections = fresh_detections + bad_detections if model_type != 'both' else \
                              self.remove_duplicates(fresh_detections, bad_detections)

            # Draw detections on original frame
            annotated_frame = self.draw_detections(frame, final_detections)

            # Encode frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                'success': True,
                'frame': f'data:image/jpeg;base64,{encoded_frame}',
                'detections': {
                    'fresh': len([d for d in final_detections if d['label'] == 'orange_fresh']),
                    'bad': len([d for d in final_detections if d['label'] == 'orange_bad']),
                    'details': final_detections
                },
                'performance': {
                    'processing_time_ms': round(processing_time, 2),
                    'metrics': self.get_performance_metrics(),
                    'system_info': self.system_info
                }
            }

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['coordinates'])
            label = det['label']
            conf = det['confidence']

            # Set color based on label (fresh=green, bad=red)
            color = (0, 255, 0) if label == 'orange_fresh' else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label with confidence
            label_text = f"{label.split('_')[1]}: {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def remove_duplicates(self, fresh_detections, bad_detections, iou_threshold=0.4):
        """Remove duplicate detections with preference for fresh oranges"""
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i < x1_i or y2_i < y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0.0

        final_detections = []
        processed_indices = set()

        # First process fresh detections
        for i, fresh_det in enumerate(fresh_detections):
            if i in processed_indices:
                continue
                
            box1 = fresh_det['coordinates']
            current_best = fresh_det
            processed_indices.add(i)
            
            # Compare with bad detections
            for j, bad_det in enumerate(bad_detections):
                box2 = bad_det['coordinates']
                iou = calculate_iou(box1, box2)
                
                # If boxes overlap significantly
                if iou > iou_threshold:
                    # Prefer fresh detection unless bad detection has significantly higher confidence
                    if bad_det['confidence'] > (fresh_det['confidence'] * 1.2):
                        current_best = bad_det
            
            final_detections.append(current_best)

        # Add remaining bad detections that don't overlap with any fresh detections
        for j, bad_det in enumerate(bad_detections):
            if j not in processed_indices:
                box1 = bad_det['coordinates']
                overlap = False
                
                for det in final_detections:
                    box2 = det['coordinates']
                    if calculate_iou(box1, box2) > iou_threshold:
                        overlap = True
                        break
                
                if not overlap:
                    final_detections.append(bad_det)

        return final_detections
