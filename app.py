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
from sklearn.ensemble import RandomForestClassifier # Changed to RandomForestClassifier

# --- OpenAI for our "real" LLM ---
import openai

app = Flask(__name__)

# --- Configuration for OpenAI ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Read from environment variable for security
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. LLM functionality will be disabled.")
    print("Please set it in your Codespaces secrets or environment variables.")
    openai_client = None
else:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

# --- Mock Ad Campaign Analytics (for LLM context) ---
# This data will be fed to the LLM as context
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
    },
    "Travel Adventures": {
        "impressions_this_week": 600000,
        "clicks_this_week": 15000,
        "ctr_this_week": 0.025,
        "budget_spent": 80000,
        "status": "Meeting Targets",
        "top_performing_ad": "Tropical Escapes",
        "underperforming_segment": "Youth demographics"
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
    ctr_base_prob = 0.02
    ctr_probs = ctr_base_prob + \
                (user_age / 1000) + \
                (np.where(ad_category == 'Tech', 0.03, 0) - np.where(ad_category == 'Fashion', 0.01, 0)) + \
                (ad_creative_freshness * 0.03) + \
                (np.where(user_genre_preference == 'SciFi', 0.02, 0)) + \
                (np.where(user_subscription_tier == 'Ad-Supported', 0.01, 0)) + \
                (np.where((user_gender == 'Male') & (user_genre_preference == 'Action'), 0.03, 0)) # Stronger Male/Action bias

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
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)) # Using RandomForestClassifier
    ])

    pipeline.fit(X, y)
    print("Mock CTR model trained successfully.")
    return pipeline, X.columns

# Initialize the ML pipeline immediately when the script runs
try:
    ml_pipeline, _ = train_mock_ctr_model()
    print("ML pipeline loaded/trained successfully at startup.")
except Exception as e:
    print(f"ERROR: Failed to initialize ML pipeline at startup: {e}")
    ml_pipeline = None # Indicate failure

# --- Flask Routes ---
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

    best_ad = None
    max_ctr = -1.0

    for ad in ADS_INVENTORY:
        input_data = pd.DataFrame([{
            'user_age': user_profile['age'],
            'user_gender': user_profile['gender'],
            'user_subscription_tier': user_profile['subscription_tier'],
            'user_watch_hours_monthly': user_profile['watch_hours'],
            'user_genre_preference': user_profile['genre_preference'],
            'ad_category': ad['category'],
            'ad_creative_freshness': ad['creative_freshness'],
            'ad_placement_time_slot': 'Evening' # Fixed for simplicity in demo
        }])

        if selected_movie_genre == 'SciFi' and ad['category'] == 'Tech':
            input_data.loc[:, 'ad_creative_freshness'] = min(1.0, input_data['ad_creative_freshness'] * 1.2)
        elif selected_movie_genre == 'Comedy' and ad['category'] == 'Food':
             input_data.loc[:, 'ad_creative_freshness'] = min(1.0, input_data['ad_creative_freshness'] * 1.1)

        predicted_proba = ml_pipeline.predict_proba(input_data)[0][1]

        if predicted_proba > max_ctr:
            max_ctr = predicted_proba
            best_ad = ad.copy()
            best_ad['predicted_ctr'] = f"{predicted_proba*100:.2f}%"

    if best_ad:
        time.sleep(0.5)
        return jsonify(best_ad)
    else:
        return jsonify({"error": "No suitable ad found"}), 404

@app.route('/ad_event', methods=['POST'])
def ad_event():
    event_data = request.json
    print(f"AD EVENT LOGGED: {event_data}")
    return jsonify({"status": "logged", "event": event_data}), 200

# --- REAL LLM Endpoint ---
@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    if openai_client is None:
        return jsonify({"answer": "LLM functionality is disabled because OPENAI_API_KEY is not set. Please check Codespaces secrets."}), 503

    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    # Convert mock analytics data to a readable string for the LLM
    analytics_context = "Current Mock Ad Campaign Analytics Data:\n\n"
    for campaign, data in MOCK_AD_CAMPAIGN_ANALYTICS.items():
        analytics_context += f"Campaign: {campaign}\n"
        for key, value in data.items():
            if isinstance(value, (int, float)):
                analytics_context += f"- {key.replace('_', ' ').title()}: {value:,}\n"
            else:
                analytics_context += f"- {key.replace('_', ' ').title()}: {value}\n"
        analytics_context += "\n"

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # You can use "gpt-4" for potentially better results but higher cost
            messages=[
                {"role": "system", "content": f"You are an expert ad campaign analyst for a streaming service. You must answer questions based ONLY on the provided analytics data. If you cannot find an answer in the data, state that you don't have enough information. Do not invent data or make assumptions. Be concise and professional. Here is the current ad campaign data:\n\n{analytics_context}"},
                {"role": "user", "content": question}
            ]
        )
        llm_response = completion.choices[0].message.content
        return jsonify({"answer": llm_response})

    except openai.APIConnectionError as e:
        print(f"OpenAI API connection error: {e}")
        return jsonify({"answer": "Could not connect to OpenAI API. Please check your internet connection or API status."}), 500
    except openai.RateLimitError as e:
        print(f"OpenAI API rate limit exceeded. Please try again shortly. {e}"), 500
        return jsonify({"answer": f"OpenAI API rate limit exceeded. Please try again shortly."}), 429 # 429 Too Many Requests
    except openai.APIStatusError as e:
        print(f"OpenAI API status error: {e}")
        return jsonify({"answer": f"OpenAI API error: {e.status_code} - {e.response.json().get('message', 'Unknown error')}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI API: {e}")
        return jsonify({"answer": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
