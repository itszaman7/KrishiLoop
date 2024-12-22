from flask import Flask, render_template,redirect,flash,Blueprint,request,jsonify
from cuba import db
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import io
from cuba.detection.orange_detector import OrangeDetector

main = Blueprint('main',__name__)

UPLOAD_FOLDER = 'cuba/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize detector once
orange_detector = OrangeDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
@main.route('/index')
def indexPage():
   context={"breadcrumb":{"parent":"Layout Light","child":"Color version"}}
   return render_template('general/index.html',**context)

@main.route("/workspace")
def workspace():
    context = {
        "breadcrumb": {
            "parent": "Workspace",
            "child": "Overview"
        }
    }
    return render_template('pages/img_detection/workspace.html', **context)

@main.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@main.route('/image-detector')
def image_detector():
    images = request.args.get('images', '').split(',')
    return render_template('pages/img_detection/ImageDetector.html', images=images)

@main.route('/detect-oranges', methods=['POST'])
def detect_oranges():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})
    
    try:
        conf_threshold = float(request.form.get('confidence', 0.25))
        image_file = request.files['image']
        
        # Use the detector class
        result = orange_detector.process_image(image_file, conf_threshold)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})