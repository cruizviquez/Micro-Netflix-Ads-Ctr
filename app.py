# app.py
from flask import Flask, render_template, request, jsonify, url_for
import random
import time
import os
import pandas as pd
import numpy as np

# --- Scikit-learn for our "real" AI model ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Our simple AI model

app = Flask(__name__)

# --- Simulated Data ---
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

# --- Mock Ad Campaign Analytics (for LLM component concept) ---
MOCK_AD_CAMPAIGN_ANALYTICS = {
    "Summer Blockbusters": {
        "impressions_this_week": 1500000,
        "clicks_this_week": 35000,
        "ctr_this_week": 0.023,
        "budget_spent": 150000,
        "status": "On Track",
        "top_performing_ad": "Pixel Paladins Game Ad",
        "underperforming_segment": "Users over 60"
    },
    "New Tech Launch": {
        "impressions_this_week": 800000,
        "clicks_this_week": 28000,
        "ctr_this_week": 0.035,
        "budget_spent": 120000,
        "status": "Exceeding Expectations",
        "top_performing_ad": "FuturePhone Pro",
        "underperforming_segment": "Late night viewers"
    }
}

# --- AI Model Training (Done at app startup for demo purposes) ---
ml_pipeline = None # Global variable to hold our trained model

def train_mock_ctr_model():
    print("Training mock CTR prediction model...")
    np.random.seed(42) # For reproducibility

    num_samples = 5000 # Smaller dataset for quick training at startup

    # Define all possible values for categorical features
    all_genders = ['Male', 'Female', 'Other']
    all_subscription_tiers = ['Premium', 'Standard', 'Ad-Supported']
    all_user_genre_preferences = ['Action', 'Comedy', 'Drama', 'SciFi', 'Kids', 'Mystery']
    all_ad_categories = ['Automotive', 'Food', 'Tech', 'Fashion', 'Travel']
    all_ad_placement_time_slots = ['Morning', 'Afternoon', 'Evening', 'Late Night']

    # Generate synthetic features for training
    user_age = np.random.randint(18, 65, num_samples)
    user_gender = np.random.choice(all_genders, num_samples)
    user_subscription_tier = np.random.choice(all_subscription_tiers, num_samples)
    user_watch_hours_monthly = np.random.normal(30, 10, num_samples).clip(0, 100)
    user_genre_preference = np.random.choice(all_user_genre_preferences, num_samples)

    ad_category = np.random.choice(all_ad_categories, num_samples)
    ad_creative_freshness = np.random.uniform(0, 1, num_samples)
    ad_placement_time_slot = np.random.choice(all_ad_placement_time_slots, num_samples)

    # Generate labels (clicks) based on some underlying "true" logic for the model to learn
    # This simulates real-world patterns that a model would detect
    ctr_base_prob = 0.02
    ctr_probs = ctr_base_prob + \
                (user_age / 1000) + \
                (np.where(ad_category == 'Tech', 0.03, 0) - np.where(ad_category == 'Fashion', 0.01, 0)) + \
                (ad_creative_freshness * 0.03) + \
                (np.where(user_genre_preference == 'SciFi', 0.02, 0)) + \
                (np.where(user_subscription_tier == 'Ad-Supported', 0.01, 0)) + \
                (np.where(user_gender == 'Female', 0.005, 0)) # Slight gender bias for demo

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

    # Define features and target
    X = train_data.drop('click', axis=1)
    y = train_data['click']

    # Define preprocessing steps
    numerical_features = ['user_age', 'user_watch_hours_monthly', 'ad_creative_freshness']
    categorical_features = ['user_gender', 'user_subscription_tier', 'user_genre_preference',
                            'ad_category', 'ad_placement_time_slot']

    # Use 'categories' parameter in OneHotEncoder to ensure consistent columns during inference
    # even if a category is missing in a single prediction request.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',
                                  categories=[all_genders, all_subscription_tiers, all_user_genre_preferences,
                                              all_ad_categories, all_ad_placement_time_slots]),
             categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though none expected here
    )

    # Create a pipeline: preprocess data then apply Logistic Regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, C=0.5)) # C is regularization strength
    ])

    # Train the pipeline
    pipeline.fit(X, y)
    print("Mock CTR model trained successfully.")
    return pipeline, X.columns # Return pipeline and expected feature columns


# --- Flask Routes ---
@app.before_first_request
def initialize_model():
    """Initializes the ML model when the Flask app first starts."""
    global ml_pipeline
    try:
        ml_pipeline, _ = train_mock_ctr_model()
        print("ML pipeline loaded/trained.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ML pipeline: {e}")
        ml_pipeline = None # Indicate failure

@app.route('/')
def index():
    return render_template('index.html', movies=MOVIES)

@app.route('/get_movies', methods=['GET'])
def get_movies():
    return jsonify(MOVIES)

@app.route('/simulate_ad_request', methods=['POST'])
def simulate_ad_request():
    global ml_pipeline
    if ml_pipeline is None:
        return jsonify({"error": "AI model not initialized. Service unavailable."}), 503

    data = request.json
    user_profile = data.get('user_profile')
    content_context = data.get('content_context')
    selected_movie_genre = content_context.get('genre')

    if not user_profile or not content_context:
        return jsonify({"error": "Missing user_profile or content_context"}), 400

    # Prepare data for prediction
    best_ad = None
    max_ctr = -1.0

    for ad in ADS_INVENTORY:
        # Construct input DataFrame for the pipeline for each ad candidate
        # IMPORTANT: Feature names and order must match training data!
        # This is where we combine user, content (movie genre), and ad features.
        input_data = pd.DataFrame([{
            'user_age': user_profile['age'],
            'user_gender': user_profile['gender'],
            'user_subscription_tier': user_profile['subscription_tier'],
            'user_watch_hours_monthly': user_profile['watch_hours'], # Adjusted from frontend
            'user_genre_preference': user_profile['genre_preference'],
            'ad_category': ad['category'],
            'ad_creative_freshness': ad['creative_freshness'],
            'ad_placement_time_slot': 'Evening' # Fixed for simplicity in demo
        }])

        # Add an influence from the content context (movie genre)
        # This is a manual feature for the demo; a real model would learn this.
        # Here we'll modify `ad_creative_freshness` based on match, for simplicity.
        if selected_movie_genre == 'SciFi' and ad['category'] == 'Tech':
            input_data.loc[:, 'ad_creative_freshness'] = min(1.0, input_data['ad_creative_freshness'] * 1.2) # Boost freshness
        elif selected_movie_genre == 'Comedy' and ad['category'] == 'Food':
             input_data.loc[:, 'ad_creative_freshness'] = min(1.0, input_data['ad_creative_freshness'] * 1.1)


        # Make prediction using the trained ML pipeline
        # predict_proba returns [prob_no_click, prob_click]
        predicted_proba = ml_pipeline.predict_proba(input_data)[0][1] # Probability of clicking

        if predicted_proba > max_ctr:
            max_ctr = predicted_proba
            best_ad = ad.copy() # Make a copy to avoid modifying original inventory
            best_ad['predicted_ctr'] = f"{predicted_proba*100:.2f}%"

    if best_ad:
        time.sleep(0.5) # Simulate a slight delay for "real-time" prediction
        return jsonify(best_ad)
    else:
        return jsonify({"error": "No suitable ad found"}), 404

@app.route('/ad_event', methods=['POST'])
def ad_event():
    event_data = request.json
    print(f"AD EVENT LOGGED: {event_data}")
    # In a real system, this would push to Kafka/data warehouse for model retraining
    return jsonify({"status": "logged", "event": event_data}), 200

# --- Mock LLM Endpoint ---
@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    question = request.json.get('question', '').lower()
    response = {"answer": "I'm a mock LLM and can only answer very specific questions about ad campaigns."}

    if "how is the ad campaign" in question or "campaign doing" in question:
        campaign_name_match = None
        for name in MOCK_AD_CAMPAIGN_ANALYTICS.keys():
            if name.lower() in question:
                campaign_name_match = name
                break

        if campaign_name_match:
            campaign_data = MOCK_AD_CAMPAIGN_ANALYTICS[campaign_name_match]
            answer_text = (
                f"The '{campaign_name_match}' campaign is currently '{campaign_data['status']}'. "
                f"This week, it has {campaign_data['impressions_this_week']:,} impressions and "
                f"{campaign_data['clicks_this_week']:,} clicks, resulting in a CTR of {campaign_data['ctr_this_week']*100:.2f}%. "
                f"Its top performing ad is '{campaign_data['top_performing_ad']}'. "
                f"An area for improvement is '{campaign_data['underperforming_segment']}'."
            )
            response["answer"] = answer_text
        else:
            response["answer"] = "Please specify which ad campaign you're asking about (e.g., 'Summer Blockbusters', 'New Tech Launch')."
    elif "hello" in question or "hi" in question:
        response["answer"] = "Hello! How can I help you with ad campaign insights?"
    elif "thank you" in question:
        response["answer"] = "You're welcome!"

    time.sleep(random.uniform(0.5, 1.5)) # Simulate LLM processing time
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
