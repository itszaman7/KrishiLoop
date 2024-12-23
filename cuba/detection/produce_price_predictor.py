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
        shelf_life = np.array([42, 28, 21, 14, 7])
        
        # Train models
        self.price_model = LinearRegression()
        self.price_model.fit(confidence_scores, prices)
        
        self.shelf_life_model = LinearRegression()
        self.shelf_life_model.fit(confidence_scores, shelf_life)
        
        print("\n=== Model Training Results ===")
        print(f"Price Model Coefficients: {self.price_model.coef_[0]:.2f}")
        print(f"Price Model Intercept: {self.price_model.intercept_:.2f}")
        print(f"Shelf Life Model Coefficients: {self.shelf_life_model.coef_[0]:.2f}")
        print(f"Shelf Life Model Intercept: {self.shelf_life_model.intercept_:.2f}\n")

    def classify_tier(self, conf):
        """Classify produce into quality tiers based on confidence score"""
        tier = None
        if conf >= 0.9:
            tier = 'S'
        elif conf >= 0.75:
            tier = 'A'
        elif conf >= 0.6:
            tier = 'B'
        elif conf >= 0.4:
            tier = 'C'
        else:
            tier = 'R'
            
        print(f"\n=== Tier Classification ===")
        print(f"Confidence Score: {conf:.3f}")
        print(f"Assigned Tier: {tier}")
        return tier

    def predict_price(self, confidence):
        """Predict price based on confidence score"""
        price = float(self.price_model.predict([[confidence]])[0])
        print(f"\n=== Price Prediction ===")
        print(f"Confidence Score: {confidence:.3f}")
        print(f"Predicted Price: {price:.2f} BDT")
        return price

    def predict_expiry(self, confidence):
        """Predict expiry date based on confidence score"""
        shelf_life = int(self.shelf_life_model.predict([[confidence]])[0])
        expiry_date = datetime.now() + timedelta(days=shelf_life)
        print(f"\n=== Expiry Prediction ===")
        print(f"Confidence Score: {confidence:.3f}")
        print(f"Predicted Shelf Life: {shelf_life} days")
        print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
        return expiry_date

    def analyze_detections(self, detections):
        """Analyze detections and provide comprehensive statistics"""
        print("\n=== Detection Analysis ===")
        print(f"Total Detections: {len(detections)}")
        
        if not detections:
            print("No detections to analyze")
            return {
                'total_count': 0,
                'tier_distribution': {},
                'price_analysis': {'average_price': 0, 'total_value': 0, 'price_range': {'min': 0, 'max': 0}},
                'expiry_analysis': {'earliest': None, 'latest': None},
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
            tier = det['tier']
            price = det['predicted_price']
            expiry = datetime.strptime(det['expiry_date'], '%Y-%m-%d')
            
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            prices.append(price)
            expiry_dates.append(expiry)
            market_recommendations.add(self.get_tier_description(tier))

        # Calculate statistics
        avg_price = np.mean(prices)
        total_value = sum(prices)
        price_range = {'min': min(prices), 'max': max(prices)}
        
        print("\nTier Distribution:")
        for tier, count in tier_distribution.items():
            print(f"Tier {tier}: {count} items")
            
        print("\nPrice Analysis:")
        print(f"Average Price: {avg_price:.2f} BDT")
        print(f"Total Value: {total_value:.2f} BDT")
        print(f"Price Range: {price_range['min']:.2f} - {price_range['max']:.2f} BDT")
        
        print("\nExpiry Analysis:")
        print(f"Earliest: {min(expiry_dates).strftime('%Y-%m-%d')}")
        print(f"Latest: {max(expiry_dates).strftime('%Y-%m-%d')}")

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
                'earliest': min(expiry_dates).strftime('%Y-%m-%d'),
                'latest': max(expiry_dates).strftime('%Y-%m-%d')
            },
            'market_recommendations': sorted(list(market_recommendations))
        }

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
