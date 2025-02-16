# Orange Quality Detection System

An AI-powered system for real-time detection and classification of orange quality using computer vision and machine learning. The system detects fresh and bad oranges, predicts prices, estimates shelf life, and provides market recommendations.

## Features

- Real-time orange detection and classification
- Quality tier assessment (S, A, B, C, R)
- Price prediction based on quality
- Shelf life estimation
- Market recommendations
- Performance monitoring and optimization
- Batch processing capabilities
- Hardware-adaptive processing

## Tech Stack

### Backend
- Python 3.8+
- Flask (Web Framework)
- SQLAlchemy (Database ORM)
- OpenCV (Image Processing)
- PyTorch (Deep Learning)
- Ultralytics YOLO (Object Detection)
- NumPy (Numerical Processing)
- Pillow (Image Handling)

### Frontend
- HTML5/CSS3
- JavaScript
- Bootstrap 5
- Chart.js (Data Visualization)

### AI/ML Components
- **Models**: 
  - YOLO (YOLOv8) for object detection
  - Dual model approach (fresh/bad orange detection)
  - Linear Regression for price prediction
  - Shelf life prediction model

- **Features**:
  - Confidence-based tier classification
  - Adaptive detection thresholds
  - IOU-based duplicate detection removal
  - Hardware-optimized inference

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. Initialize database:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Download model weights:
```bash
mkdir -p models/good models/bad
# Place fresh_oranges.pt in models/good/
# Place bad_oranges.pt in models/bad/
```

6. Run the application:
```bash
python run.py
```

## System Requirements

### Minimum Requirements
- CPU: Intel Core i5 or equivalent
- RAM: 8GB
- Storage: 2GB free space
- Python 3.8+

### Recommended Requirements
- CPU: Intel Core i7 or equivalent
- RAM: 16GB
- GPU: NVIDIA GPU with CUDA support
- Storage: 5GB free space

## Performance Optimization

The system automatically adapts to available hardware:

### GPU Mode
- Input Size: 640x640
- Confidence Threshold: 0.25
- Scale Percent: 75%
- JPEG Quality: 85%

### CPU Mode
- Input Size: 416x416
- Confidence Threshold: 0.35
- Scale Percent: 50%
- JPEG Quality: 75%

## Project Structure
```
cuba/
├── detection/
│   ├── orange_detector.py    # Main detection logic
│   ├── realtime.py          # Real-time processing
│   └── produce_price_predictor.py  # Price prediction
├── templates/
│   └── pages/
│       └── img_detection/   # Frontend templates
├── static/
│   └── uploads/            # Image upload directory
├── models/
│   ├── good/              # Fresh orange model
│   └── bad/               # Bad orange model
└── routes.py              # API endpoints
```

## API Endpoints

- `/detect-oranges`: Process uploaded images
- `/realtime-detect`: Real-time video processing
- `/save-detection`: Save detection results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
