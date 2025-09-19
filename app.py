# app.py - Complete File
from flask import Flask, render_template, request, jsonify, url_for
import random
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

app = Flask(__name__)

# --- Mock Data ---
MOVIES = [
    {"id": "mv1", "title": "The Cosmic Odyssey", "genre": "SciFi", "thumbnail": "https://via.placeholder.com/200x120?text=SciFi"},
    {"id": "mv2", "title": "Laughing Gas", "genre": "Comedy", "thumbnail": "https://via.placeholder.com/200x120?text=Comedy"},
    {"id": "mv3", "title": "Heartland Echoes", "genre": "Drama", "thumbnail": "https://via.placeholder.com/200x120?text=Drama"},
    {"id": "mv4", "title": "Pixel Paladins", "genre": "Action", "thumbnail": "https://via.placeholder.com/200x120?text=Action"},
    {"id": "mv5", "title": "Whispering Woods", "genre": "Mystery", "thumbnail": "https://via.placeholder.com/200x120?text=Mystery"},
]

ADS_INVENTORY = [
    {"id": "ad1", "title": "New Quantum Car", "category": "Automotive", "creative_freshness": 0.8, "image": "https://via.placeholder.com/640x360?text=Quantum+Car+Ad"},
    {"id": "ad2", "title": "Delicious Snack Bites", "category": "Food", "creative_freshness": 0.9, "image": "https://via.placeholder.com/640x360?text=Snack+Ad"},
    {"id": "ad3", "title": "FuturePhone Pro", "category": "Tech", "creative_freshness": 0.7, "image": "https://via.placeholder.com/640x360?text=FuturePhone+Ad"},
    {"id": "ad4", "title": "Fashion Forward", "category": "Fashion", "creative_freshness": 0.6, "image": "https://via.placeholder.com/640x360?text=Fashion+Ad"},
    {"id": "ad5", "title": "Tropical Escapes", "category": "Travel", "creative_freshness": 0.5, "image": "https://via.placeholder.com/640x360?text=Travel+Ad"},
]

# --- Mock LLM Class ---
class MockLLM:
    def generate_ad_copy(self, user_profile):
        templates = {
            'young_mobile': [
                "🎬 Binge your favorites on-the-go! 50% off mobile plan",
                "📱 Stream anywhere, anytime. Student discount available!"
            ],
            'adult_tv': [
                "🏠 Family movie night? Get 4K streaming for your Smart TV",
                "Premium content, zero ads. Free trial for 30 days!"
            ],
            'senior_desktop': [
                "Easy streaming, classic content. Simple interface, great value",
                "Discover timeless movies and shows. Easy setup guaranteed"
            ],
            'default': [
                "Personalized entertainment just for you. Start free trial",
                "Thousands of titles waiting. Join millions of happy viewers"
            ]
        }
        
        age = user_profile.get('age', 30)
        device = user_profile.get('device', 'desktop').lower()
        
        if age < 30 and 'mobile' in device:
            segment = 'young_mobile'
        elif age >= 50:
            segment = 'senior_desktop'
        elif 'tv' in device:
            segment = 'adult_tv'
        else:
            segment = 'default'
            
        return random.choice(templates[segment])
    
    def analyze_performance(self, ad_text, ctr):
        if ctr > 0.05:
            return f"High performing ad! CTR: {ctr:.1%}. Key factors: engaging copy, right audience match"
        else:
            return f"Consider A/B testing different copy. Current CTR: {ctr:.1%}"

# --- ML Model Training ---
def train_mock_ctr_model():
    print("Training mock CTR prediction model...")
    np.random.seed(42)

    num_samples = 5000

    # Define all possible values for categorical features
    all_genders = ['Male', 'Female', 'Other']
    all_subscription_tiers = ['Premium', 'Standard', 'Ad-Supported']
    all_user_genre_preferences = ['Action', 'Comedy', 'Drama', 'SciFi', 'Kids', 'Mystery']
    all_ad_categories = ['Automotive', 'Food', 'Tech', 'Fashion', 'Travel']
    all_ad_placement_time_slots = ['Morning', 'Afternoon', 'Evening', 'Late Night']

    # Generate synthetic features
    user_age = np.random.randint(18, 65, num_samples)
    user_gender = np.random.choice(all_genders, num_samples)
    user_subscription_tier = np.random.choice(all_subscription_tiers, num_samples)
    user_watch_hours_monthly = np.random.normal(30, 10, num_samples).clip(0, 100)
    user_genre_preference = np.random.choice(all_user_genre_preferences, num_samples)
    ad_category = np.random.choice(all_ad_categories, num_samples)
    ad_creative_freshness = np.random.uniform(0, 1, num_samples)
    ad_placement_time_slot = np.random.choice(all_ad_placement_time_slots, num_samples)

    # Generate labels
    ctr_base_prob = 0.02
    ctr_probs = ctr_base_prob + \
                (user_age / 1000) + \
                (np.where(ad_category == 'Tech', 0.03, 0) - np.where(ad_category == 'Fashion', 0.01, 0)) + \
                (ad_creative_freshness * 0.03) + \
                (np.where(user_genre_preference == 'SciFi', 0.02, 0)) + \
                (np.where(user_subscription_tier == 'Ad-Supported', 0.01, 0)) + \
                (np.where((user_gender == 'Male') & (user_genre_preference == 'Action'), 0.03, 0))

    ctr_probs = np.clip(ctr_probs, 0.001, 0.1)
    clicks = (np.random.rand(num_samples) < ctr_probs).astype(int)

    train_data = pd.DataFrame({
        'user_age': user_age,
        'user_gender': user_gender,
        'user_subscription_tier': user_subscription_tier,
        'user_watch_hours_monthly': user_watch_hours_monthly,
        'user_genre_preference': user_genre_preference,
        'ad_category': ad_category,
        'ad_creative_freshness': ad_creative_freshness,
        'ad_placement_time_slot': ad_placement_time_slot,
        'click': clicks
    })

    X = train_data.drop('click', axis=1)
    y = train_data['click']

    numerical_features = ['user_age', 'user_watch_hours_monthly', 'ad_creative_freshness']
    categorical_features = ['user_gender', 'user_subscription_tier', 'user_genre_preference',
                            'ad_category', 'ad_placement_time_slot']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',
                                  categories=[all_genders, all_subscription_tiers, all_user_genre_preferences,
                                              all_ad_categories, all_ad_placement_time_slots]),
             categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])

    pipeline.fit(X, y)
    print("Mock CTR model trained successfully.")
    return pipeline, X.columns

# --- Ad Optimizer Class ---
class AdOptimizer:
    def __init__(self, ml_pipeline):
        self.ml_pipeline = ml_pipeline
        self.llm = MockLLM()
        self.performance_history = []
    
    def select_best_ad(self, user_profile, content_context, ads_inventory):
        best_ad = None
        max_score = -1.0
        predictions = []
        
        for ad in ads_inventory:
            input_data = self._prepare_ml_input(user_profile, ad, content_context)
            predicted_ctr = self.ml_pipeline.predict_proba(input_data)[0][1]
            score = self._calculate_ad_score(predicted_ctr, user_profile, ad, content_context)
            
            predictions.append({
                'ad': ad,
                'predicted_ctr': predicted_ctr,
                'score': score
            })
            
            if score > max_score:
                max_score = score
                best_ad = ad.copy()
                best_ad['predicted_ctr'] = predicted_ctr
                best_ad['score'] = score
        
        if best_ad:
            best_ad['personalized_copy'] = self.llm.generate_ad_copy(user_profile)
            best_ad['insights'] = self.llm.analyze_performance(
                best_ad['personalized_copy'], 
                best_ad['predicted_ctr']
            )
        
        return best_ad, predictions
    
    def _prepare_ml_input(self, user_profile, ad, content_context):
        return pd.DataFrame([{
            'user_age': user_profile.get('age', 30),
            'user_gender': user_profile.get('gender', 'Other'),
            'user_subscription_tier': user_profile.get('subscription_tier', 'Standard'),
            'user_watch_hours_monthly': user_profile.get('watch_hours', 20),
            'user_genre_preference': user_profile.get('genre_preference', 'Drama'),
            'ad_category': ad['category'],
            'ad_creative_freshness': ad['creative_freshness'],
            'ad_placement_time_slot': self._get_time_slot()
        }])
    
    def _calculate_ad_score(self, predicted_ctr, user_profile, ad, content_context):
        score = predicted_ctr
        
        if user_profile.get('genre_preference') == content_context.get('genre'):
            score *= 1.2
        
        score *= (1 + ad['creative_freshness'] * 0.1)
        
        hour = datetime.now().hour
        if 19 <= hour <= 22:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _get_time_slot(self):
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 23:
            return 'Evening'
        else:
            return 'Late Night'

# Initialize components
try:
    ml_pipeline, _ = train_mock_ctr_model()
    ad_optimizer = AdOptimizer(ml_pipeline)
    print("All components initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize components: {e}")
    ml_pipeline = None
    ad_optimizer = None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', movies=MOVIES)

@app.route('/get_movies', methods=['GET'])
def get_movies():
    return jsonify(MOVIES)

@app.route('/simulate_ad_request', methods=['POST'])
def simulate_ad_request():
    if ad_optimizer is None:
        return jsonify({"error": "AI model not initialized"}), 503

    data = request.json
    user_profile = data.get('user_profile')
    content_context = data.get('content_context')

    if not user_profile or not content_context:
        return jsonify({"error": "Missing required data"}), 400

    best_ad, all_predictions = ad_optimizer.select_best_ad(
        user_profile, 
        content_context, 
        ADS_INVENTORY
    )

    if best_ad:
        time.sleep(0.5)  # Simulate processing
        return jsonify(best_ad)
    else:
                return jsonify({"error": "No suitable ad found"}), 404

@app.route('/api/optimize_ad', methods=['POST'])
def optimize_ad():
    """Main API endpoint for ad optimization"""
    try:
        data = request.json
        user_profile = data.get('user_profile')
        content_context = data.get('content_context')
        
        if not user_profile or not content_context:
            return jsonify({"error": "Missing required data"}), 400
        
        # Get optimized ad
        best_ad, all_predictions = ad_optimizer.select_best_ad(
            user_profile, 
            content_context, 
            ADS_INVENTORY
        )
        
        if best_ad:
            # Log the impression
            log_impression(user_profile, best_ad, content_context)
            
            return jsonify({
                "success": True,
                "ad": best_ad,
                "alternatives": all_predictions[:3]  # Top 3 alternatives
            })
        else:
            return jsonify({"error": "No suitable ad found"}), 404
            
    except Exception as e:
        app.logger.error(f"Error in optimize_ad: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/ad_event', methods=['POST'])
def ad_event():
    """Log ad events (impressions, clicks, etc.)"""
    event_data = request.json
    print(f"AD EVENT LOGGED: {event_data}")
    log_event(event_data)
    return jsonify({"status": "logged", "event": event_data}), 200

@app.route('/api/report_click', methods=['POST'])
def report_click():
    """Track ad clicks for learning"""
    data = request.json
    log_click_event(data)
    return jsonify({"status": "recorded"})

@app.route('/test', methods=['GET'])
def test_page():
    """Test page for development"""
    sample_user = {
        'age': 28,
        'gender': 'Male',
        'subscription_tier': 'Standard',
        'watch_hours': 35,
        'genre_preference': 'Action',
        'device': 'mobile'
    }
    return render_template('test.html', sample_user=sample_user, movies=MOVIES)

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    metrics = calculate_metrics()
    return render_template('dashboard.html', metrics=metrics)

@app.route('/api/test_llm', methods=['GET'])
def test_llm():
    """Test LLM functionality"""
    test_user = {
        'age': 28,
        'device': 'mobile',
        'watch_hours': 4.5,
        'genre_preference': 'Action'
    }
    
    llm = MockLLM()
    ad_copy = llm.generate_ad_copy(test_user)
    
    return jsonify({
        'test_user': test_user,
        'generated_ad': ad_copy,
        'status': 'LLM working correctly'
    })

@app.route('/api/generate_test_data', methods=['GET'])
def generate_test_data():
    """Generate sample CTR data for testing"""
    try:
        df = generate_sample_data(1000)
        return jsonify({
            'success': True,
            'samples_generated': len(df),
            'overall_ctr': f"{df['clicked'].mean():.2%}",
            'data_preview': df.head(5).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Helper Functions ---
def log_impression(user_profile, ad, context):
    """Log ad impression for analysis"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'event_type': 'impression',
        'user': user_profile,
        'ad_id': ad.get('id'),
        'ad_title': ad.get('title'),
        'predicted_ctr': ad.get('predicted_ctr'),
        'context': context
    }
    app.logger.info(f"Impression: {log_entry}")

def log_event(event_data):
    """Generic event logging"""
    timestamp = datetime.now().isoformat()
    event_data['timestamp'] = timestamp
    app.logger.info(f"Event: {event_data}")

def log_click_event(data):
    """Log click events"""
    timestamp = datetime.now().isoformat()
    click_data = {
        'timestamp': timestamp,
        'event_type': 'click',
        'ad_id': data.get('ad_id'),
        'user_id': data.get('user_id'),
        'session_id': data.get('session_id')
    }
    app.logger.info(f"Click: {click_data}")

def calculate_metrics():
    """Calculate performance metrics for dashboard"""
    # In production, these would come from a database
    return {
        'total_impressions': random.randint(10000, 50000),
        'total_clicks': random.randint(500, 2500),
        'avg_ctr': f"{random.uniform(0.02, 0.08):.2%}",
        'revenue': f"${random.randint(5000, 25000):,}",
        'top_performing_ads': [
            {'name': 'FuturePhone Pro', 'ctr': '7.2%'},
            {'name': 'Quantum Car', 'ctr': '5.8%'},
            {'name': 'Snack Bites', 'ctr': '4.9%'}
        ],
        'hourly_performance': generate_hourly_data()
    }

def generate_hourly_data():
    """Generate mock hourly performance data"""
    hours = list(range(24))
    return {
        'hours': hours,
        'impressions': [random.randint(100, 1000) for _ in hours],
        'clicks': [random.randint(5, 50) for _ in hours]
    }

def generate_sample_data(n_samples=1000):
    """Generate sample CTR data for testing"""
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        age = np.random.randint(18, 70)
        gender = np.random.choice(['Male', 'Female', 'Other'])
        device = np.random.choice(['mobile', 'desktop', 'smart_tv', 'tablet'])
        watch_time_hours = np.random.exponential(2.5)
        preferred_genre = np.random.choice(['Action', 'Comedy', 'Drama', 'SciFi', 'Mystery'])
        
        # Simulate CTR logic
        base_ctr = 0.02
        if 25 <= age <= 45:
            base_ctr *= 1.5
        if device == 'smart_tv':
            base_ctr *= 1.3
        
        ctr = base_ctr * np.random.uniform(0.7, 1.3)
        clicked = 1 if np.random.random() < ctr else 0
        
        data.append({
            'user_id': f'user_{_}',
            'age': age,
            'gender': gender,
            'device': device,
            'watch_time_hours': round(watch_time_hours, 2),
            'preferred_genre': preferred_genre,
            'clicked': clicked,
            'actual_ctr': round(ctr, 4)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/sample_ctr_data.csv', index=False)
    return df

# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)