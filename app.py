# app.py
from flask import Flask, render_template, request, jsonify, url_for
import random
import time
import os
import pandas as pd
import numpy as np

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

# --- Mock CTR Prediction Logic (This is the "ML model" for the demo) ---
def mock_predict_ctr(user_profile, content_context, ad):
    base_ctr = 0.02 # Default low CTR

    # Factors influencing CTR (simplified for demo)
    # User Age: Older users might click on Automotive/Travel more
    if user_profile['age'] > 40 and ad['category'] in ['Automotive', 'Travel']:
        base_ctr += 0.015
    # User Genre Preference matches Ad Category
    if user_profile['genre_preference'] == 'SciFi' and ad['category'] == 'Tech':
        base_ctr += 0.02
    if user_profile['genre_preference'] == 'Comedy' and ad['category'] == 'Food':
        base_ctr += 0.01
    # Ad-Supported Tier users might be more prone to click
    if user_profile['subscription_tier'] == 'Ad-Supported':
        base_ctr += 0.005
    # Creative Freshness: Newer ads might have higher CTR
    base_ctr += ad['creative_freshness'] * 0.01

    # Content Context: Tech ads during SciFi content
    if content_context['genre'] == 'SciFi' and ad['category'] == 'Tech':
        base_ctr += 0.02

    # Random noise to make it slightly dynamic
    base_ctr += random.uniform(-0.005, 0.005)

    return max(0.001, min(0.1, base_ctr)) # Keep CTR between 0.1% and 10%

# --- Mock Ad Campaign Analytics (for LLM component concept) ---
# In a real scenario, this would come from a data warehouse
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


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', movies=MOVIES)

@app.route('/get_movies', methods=['GET'])
def get_movies():
    return jsonify(MOVIES)

@app.route('/simulate_ad_request', methods=['POST'])
def simulate_ad_request():
    data = request.json
    user_profile = data.get('user_profile')
    content_context = data.get('content_context')

    if not user_profile or not content_context:
        return jsonify({"error": "Missing user_profile or content_context"}), 400

    # Simulate real-time prediction by iterating through ads and picking the best
    best_ad = None
    max_ctr = -1.0
    for ad in ADS_INVENTORY:
        predicted_ctr = mock_predict_ctr(user_profile, content_context, ad)
        if predicted_ctr > max_ctr:
            max_ctr = predicted_ctr
            best_ad = ad
            best_ad['predicted_ctr'] = f"{predicted_ctr*100:.2f}%"

    if best_ad:
        # Simulate a slight delay for "real-time" prediction
        time.sleep(0.5)
        return jsonify(best_ad)
    else:
        return jsonify({"error": "No suitable ad found"}), 404

@app.route('/ad_event', methods=['POST'])
def ad_event():
    event_data = request.json
    print(f"AD EVENT LOGGED: {event_data}")
    # In a real system, this would push to Kafka/data warehouse
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
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is fine for demo
