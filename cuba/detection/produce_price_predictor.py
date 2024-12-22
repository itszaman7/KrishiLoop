import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class ProducePricePredictor:
    def __init__(self):
        self.price_model = None
        self.shelf_life_model = None
        self.setup_models()

    def setup_models(self):
        """Initialize and train the models with the provided dataset"""
        # Training data for price prediction
        confidence_scores = np.array([0.95, 0.85, 0.65, 0.45, 0.30]).reshape(-1, 1)
        prices = np.array([400, 350, 300, 250, 200])
        
        # Training data for shelf life (days)
        # S tier: 6 weeks (42 days) - Premium storage & handling
        # A tier: 4 weeks (28 days) - Refrigerated storage
        # B tier: 3 weeks (21 days) - Standard storage
        # C tier: 2 weeks (14 days) - Limited storage
        # R tier: 1 week (7 days) - Immediate processing needed
        shelf_life = np.array([42, 28, 21, 14, 7])
        
        # Train models
        self.price_model = LinearRegression()
        self.price_model.fit(confidence_scores, prices)
        
        self.shelf_life_model = LinearRegression()
        self.shelf_life_model.fit(confidence_scores, shelf_life)

    def classify_tier(self, conf):
        """
        Classify produce into quality tiers based on confidence score
        
        Args:
            conf (float): Confidence score from the detection model
            
        Returns:
            str: Quality tier (S, A, B, C, or R)
        """
        if conf >= 0.9:
            return 'S'  # Export Quality (6 weeks shelf life)
        elif conf >= 0.75:
            return 'A'  # Supermarket Quality (4 weeks shelf life)
        elif conf >= 0.6:
            return 'B'  # Local Market Quality (3 weeks shelf life)
        elif conf >= 0.4:
            return 'C'  # Community Market Quality (2 weeks shelf life)
        else:
            return 'R'  # Recycling Grade (1 week or less)

    def predict_price(self, confidence):
        """
        Predict price based on confidence score using linear regression
        
        Args:
            confidence (float): Confidence score from the detection model
            
        Returns:
            float: Predicted price in BDT
        """
        return float(self.price_model.predict([[confidence]])[0])

    def predict_expiry(self, confidence):
        """
        Predict expiry date based on confidence score and storage conditions
        
        Args:
            confidence (float): Confidence score from the detection model
            
        Returns:
            datetime: Predicted expiry date
        """
        shelf_life = int(self.shelf_life_model.predict([[confidence]])[0])
        return datetime.now() + timedelta(days=shelf_life)

    def get_tier_description(self, tier):
        """
        Get market description and storage recommendations for each tier
        
        Args:
            tier (str): Quality tier
            
        Returns:
            str: Market description with storage recommendations
        """
        descriptions = {
            'S': 'Export Quality - International Markets (UAE, UK, Germany)\n'
                'Storage: Refrigerated at 2-4°C, 85-90% humidity. Shelf life: 6 weeks',
            'A': 'Supermarket Quality - Premium Local Stores (Shwapno, Meena Bazar, Agora)\n'
                'Storage: Refrigerated at 4-7°C, 85% humidity. Shelf life: 4 weeks',
            'B': 'Local Market Quality - Standard Retail Markets\n'
                'Storage: Cool, dry place at 10-15°C. Shelf life: 3 weeks',
            'C': 'Community Market Quality - Affordable Access\n'
                'Storage: Room temperature, consume within 2 weeks',
            'R': 'Recycling Grade - Organic Fertilizer Production\n'
                'Process within 1 week for optimal resource recovery'
        }
        return descriptions.get(tier, 'Unknown Tier')

    def analyze_detections(self, detections):
        """
        Analyze detections and provide comprehensive statistics
        
        Args:
            detections (list): List of detection objects with confidence scores
            
        Returns:
            dict: Analysis results including tier distribution, prices, and market recommendations
        """
        if not detections:
            return {
                'total_count': 0,
                'tier_distribution': {},
                'price_analysis': {
                    'average_price': 0,
                    'total_value': 0,
                    'price_range': {'min': 0, 'max': 0}
                },
                'expiry_analysis': {
                    'earliest': None,
                    'latest': None
                },
                'market_recommendations': []
            }

        # Initialize analysis
        tier_distribution = {}
        prices = []
        expiry_dates = []
        market_recommendations = set()

        # Process each detection
        for det in detections:
            confidence = det['confidence']
            tier = self.classify_tier(confidence)
            price = self.predict_price(confidence)
            expiry = self.predict_expiry(confidence)
            
            # Update distributions
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            prices.append(price)
            expiry_dates.append(expiry)
            
            # Add market recommendation
            market_recommendations.add(self.get_tier_description(tier))

        # Calculate statistics
        avg_price = np.mean(prices)
        total_value = sum(prices)
        price_range = {'min': min(prices), 'max': max(prices)}
        
        # Format expiry dates
        earliest_expiry = min(expiry_dates)
        latest_expiry = max(expiry_dates)

        return {
            'total_count': len(detections),
            'tier_distribution': tier_distribution,
            'price_analysis': {
                'average_price': round(avg_price, 2),
                'total_value': round(total_value, 2),
                'price_range': {
                    'min': round(price_range['min'], 2),
                    'max': round(price_range['max'], 2)
                }
            },
            'expiry_analysis': {
                'earliest': earliest_expiry.strftime('%Y-%m-%d'),
                'latest': latest_expiry.strftime('%Y-%m-%d')
            },
            'market_recommendations': sorted(list(market_recommendations))
        }
