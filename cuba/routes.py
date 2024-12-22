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
from cuba.models import Batch, Stock, Produce, Sale, Detection
from datetime import datetime
import builtins  # For getattr function
import time

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
    # Get all active batches
    batches = Batch.query.filter_by(status='active').all()
    
    context = {
        "breadcrumb": {
            "parent": "Image Detection",
            "child": "Workspace"
        },
        "batches": batches
    }
    
    return render_template('pages/img_detection/workspace.html', **context)

@main.route('/batch', methods=['GET'])
def batch():
    batches = Batch.query.order_by(Batch.created_at.desc()).all()
    
    # Count batches that have items in each tier
    batches_by_counts = {
        's': sum(1 for b in batches if b.tier_s_count > 0),
        'a': sum(1 for b in batches if b.tier_a_count > 0),
        'b': sum(1 for b in batches if b.tier_b_count > 0),
        'c': sum(1 for b in batches if b.tier_c_count > 0),
        'r': sum(1 for b in batches if b.tier_r_count > 0),
    }
    
    context = {
        "breadcrumb": {
            "parent": "Batch Manager",
            "child": "Overview"
        },
        "batches": batches,
        "batches_by_counts": batches_by_counts,
        "getattr": getattr  # Pass getattr function to template
    }
    return render_template('pages/batch/batchManager.html', **context)

@main.route('/batch/create', methods=['POST'])
def create_batch_form():
    try:
        name = request.form.get('name')
        description = request.form.get('description', '')
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'Batch name is required'
            })
            
        new_batch = Batch(name=name, description=description)
        db.session.add(new_batch)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'id': new_batch.id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            })

        file = request.files['image']
        batch_id = request.form.get('batch_id')

        if not batch_id:
            return jsonify({
                'success': False,
                'error': 'No batch selected'
            })

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to filename to prevent duplicates
            filename = f"{int(time.time())}_{filename}"
            
            # Ensure upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            return jsonify({
                'success': True,
                'filename': filename
            })

        return jsonify({
            'success': False,
            'error': 'Invalid file type'
        })

    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main.route('/image-detector')
def image_detector():
    images = request.args.get('images', '').split(',')
    return render_template('pages/img_detection/ImageDetector.html', images=images)

@main.route('/detect-oranges', methods=['POST'])
def detect_oranges():
    try:
      
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            })

        image_file = request.files['image']
        batch_id = request.form.get('batch_id')
        conf_threshold = float(request.form.get('confidence_threshold', 0.25))

        # Process the image
        results = orange_detector.process_image(image_file, conf_threshold)
        
        # Debug print
        print("Detection results:", {
            'success': results['success'],
            'has_fresh_image': 'fresh_image' in results,
            'has_bad_image': 'bad_image' in results
        })

        if not results['success']:
            return jsonify({
                'success': False,
                'error': results['error']
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error in detect_oranges: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# API Routes for Batch Manager
@main.route('/api/batch/create', methods=['POST'])
def create_batch_api():
    try:
        new_batch = Batch(
            name=request.form['name'],
            description=request.form.get('description', '')
        )
        db.session.add(new_batch)
        db.session.commit()
        return jsonify({'success': True, 'id': new_batch.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@main.route('/api/batch/<int:batch_id>', methods=['GET'])
def get_batch(batch_id):
    try:
        batch = Batch.query.get_or_404(batch_id)
        return jsonify({
            'success': True,
            'data': {
                'id': batch.id,
                'name': batch.name,
                'created_at': batch.created_at.strftime('%Y-%m-%d %H:%M'),
                'status': batch.status,
                'total_items': len(batch.produce_items) if batch.produce_items else 0,
                'items_sold': sum(1 for item in batch.produce_items if item.status == 'sold') if batch.produce_items else 0,
                'total_sales': sum(sale.total_price for sale in batch.sales) if batch.sales else 0,
                'stocks': [{
                    'id': stock.id,
                    'tier': stock.tier,
                    'quantity': stock.quantity,
                    'price_per_unit': float(stock.price_per_unit),
                    'total_price': float(stock.total_price),
                    'status': stock.status,
                    'expiry_date': stock.expiry_date.strftime('%Y-%m-%d') if stock.expiry_date else None
                } for stock in batch.stocks]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/api/batch/<int:batch_id>', methods=['DELETE'])
def delete_batch(batch_id):
    try:
        batch = Batch.query.get_or_404(batch_id)
        db.session.delete(batch)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@main.route('/api/batch/<int:batch_id>/items')
def get_batch_items(batch_id):
    try:
        batch = Batch.query.get_or_404(batch_id)
        items = Produce.query.filter_by(batch_id=batch_id).all()
        
        return jsonify({
            'success': True,
            'items': [{
                'id': item.id,
                'image_path': item.image_path,
                'confidence': item.confidence,
                'tier': item.tier,
                'price': item.price,
                'expiry_date': item.expiry_date.isoformat(),
            } for item in items]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main.route('/save-detection-results', methods=['POST'])
def save_detection_results():
    try:
        print("Received save request")
        data = request.json
        print("Request data:", data)

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            })

        batch_id = data.get('batch_id')
        if not batch_id:
            return jsonify({
                'success': False,
                'error': 'No batch selected'
            })

        # Get the batch
        batch = Batch.query.get(batch_id)
        if not batch:
            return jsonify({
                'success': False,
                'error': f'Invalid batch ID: {batch_id}'
            })

        # Create Produce items from detections
        try:
            # Process fresh detections
            for detection in data['fresh_detections']:
                produce = Produce(
                    batch_id=batch_id,
                    confidence=detection['confidence'],
                    tier=detection['tier'],
                    price=detection['predicted_price'],
                    expiry_date=datetime.strptime(detection['expiry_date'], '%Y-%m-%d'),
                    market_recommendation=detection['market_recommendation'],
                    x1=detection['coordinates'][0],
                    y1=detection['coordinates'][1],
                    x2=detection['coordinates'][2],
                    y2=detection['coordinates'][3]
                )
                db.session.add(produce)

            # Process bad detections
            for detection in data['bad_detections']:
                produce = Produce(
                    batch_id=batch_id,
                    confidence=detection['confidence'],
                    tier=detection['tier'],
                    price=detection['predicted_price'],
                    expiry_date=datetime.strptime(detection['expiry_date'], '%Y-%m-%d'),
                    market_recommendation=detection['market_recommendation'],
                    x1=detection['coordinates'][0],
                    y1=detection['coordinates'][1],
                    x2=detection['coordinates'][2],
                    y2=detection['coordinates'][3]
                )
                db.session.add(produce)

            # Update batch analysis
            batch.update_analysis()
            
            db.session.commit()
            print("Results saved successfully to batch")

            return jsonify({
                'success': True,
                'message': 'Results saved successfully'
            })

        except Exception as e:
            db.session.rollback()
            print(f"Database error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Database error: {str(e)}'
            })

    except Exception as e:
        print(f"Error in save_detection_results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })