from . import db
from datetime import datetime
from flask_login import UserMixin


class User(db.Model,UserMixin):
    id = db.Column(db.Integer,primary_key = True)
    username = db.Column(db.String(20),unique=True,nullable=False)
    email = db.Column(db.String(120),unique=True,nullable=False)
    password = db.Column(db.String(600),nullable=False)
    isAdmin = db.Column(db.Boolean,default=False)

    def __repr__(self):
        return f"User('{self.username}','{self.email}')" 

class Todo(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    description = db.Column(db.String(500),unique=True,nullable=False)
    completed = db.Column(db.Boolean,default=False)
    timeStamp = db.Column(db.DateTime,default=datetime.utcnow) 

class Batch(db.Model):
    """A batch of produce items (e.g., oranges)"""
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    status = db.Column(db.String(20), default='active')  # active, partially_sold, sold_out, archived
    
    # Relationships
    produce_items = db.relationship('Produce', backref='batch', lazy=True)
    stocks = db.relationship('Stock', backref='batch', lazy=True)
    sales = db.relationship('Sale', backref='batch', lazy=True)
    
    # Analysis fields
    total_price = db.Column(db.Float, default=0.0)
    average_price = db.Column(db.Float, default=0.0)
    total_items = db.Column(db.Integer, default=0)
    
    # Sales tracking
    items_sold = db.Column(db.Integer, default=0)
    total_sales = db.Column(db.Float, default=0.0)  # Total money earned
    average_sale_price = db.Column(db.Float, default=0.0)
    
    # Tier distribution
    tier_s_count = db.Column(db.Integer, default=0)
    tier_a_count = db.Column(db.Integer, default=0)
    tier_b_count = db.Column(db.Integer, default=0)
    tier_c_count = db.Column(db.Integer, default=0)
    tier_r_count = db.Column(db.Integer, default=0)
    
    def update_analysis(self):
        """Update batch analysis based on produce items and sales"""
        # Reset produce counters
        self.total_items = len(self.produce_items)
        self.total_price = sum(item.price for item in self.produce_items)
        self.average_price = self.total_price / self.total_items if self.total_items > 0 else 0
        
        # Reset tier counts
        self.tier_s_count = sum(1 for item in self.produce_items if item.tier == 'S')
        self.tier_a_count = sum(1 for item in self.produce_items if item.tier == 'A')
        self.tier_b_count = sum(1 for item in self.produce_items if item.tier == 'B')
        self.tier_c_count = sum(1 for item in self.produce_items if item.tier == 'C')
        self.tier_r_count = sum(1 for item in self.produce_items if item.tier == 'R')
        
        # Update sales analysis
        self.items_sold = sum(sale.quantity for sale in self.sales)
        self.total_sales = sum(sale.total_price for sale in self.sales)
        self.average_sale_price = self.total_sales / self.items_sold if self.items_sold > 0 else 0
        
        # Update batch status based on sales
        if self.items_sold == 0:
            self.status = 'active'
        elif self.items_sold < self.total_items:
            self.status = 'partially_sold'
        else:
            self.status = 'sold_out'
        
        db.session.commit()

class Produce(db.Model):
    """Individual produce item (e.g., single orange)"""
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=True)
    
    # Detection data
    image_path = db.Column(db.String(500))
    confidence = db.Column(db.Float, nullable=False)
    tier = db.Column(db.String(1), nullable=False)  # S, A, B, C, R
    price = db.Column(db.Float, nullable=False)
    expiry_date = db.Column(db.DateTime, nullable=False)
    
    # Market recommendation
    market_recommendation = db.Column(db.String(500))
    
    # Coordinates from detection
    x1 = db.Column(db.Integer)
    y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer)
    y2 = db.Column(db.Integer)

class Stock(db.Model):
    """A stock of produce items (e.g., crate of 64 oranges)"""
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Stock details
    quantity = db.Column(db.Integer, default=64)
    tier = db.Column(db.String(1), nullable=False)  # Single tier per stock
    price_per_unit = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    expiry_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='available')  # available, reserved, sold
    
    # Relationships
    produce_items = db.relationship('Produce', backref='stock', lazy=True)
    sales = db.relationship('Sale', backref='stock', lazy=True)

class Sale(db.Model):
    """Record of a stock sale"""
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=False)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Sale details
    quantity = db.Column(db.Integer, nullable=False)
    price_per_unit = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    tier = db.Column(db.String(1), nullable=False)
    
    # Additional info
    payment_status = db.Column(db.String(20), default='pending')  # pending, completed, cancelled
    notes = db.Column(db.String(500))
    
    def __init__(self, batch_id, stock_id, buyer_id, quantity, price_per_unit, tier):
        """Initialize a sale"""
        self.batch_id = batch_id
        self.stock_id = stock_id
        self.buyer_id = buyer_id
        self.quantity = quantity
        self.price_per_unit = price_per_unit
        self.total_price = price_per_unit * quantity
        self.tier = tier

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    fresh_count = db.Column(db.Integer, default=0)
    bad_count = db.Column(db.Integer, default=0)
    fresh_analysis = db.Column(db.JSON)
    bad_analysis = db.Column(db.JSON)
    fresh_detections = db.Column(db.JSON)
    bad_detections = db.Column(db.JSON)

    batch = db.relationship('Batch', backref=db.backref('detections', lazy=True))