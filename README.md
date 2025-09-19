# Micro Netflix Ads CTR Optimizer

A Netflix-style streaming platform demo showcasing real-time ad optimization using Machine Learning to maximize Click-Through Rates (CTR).

## Overview

This project demonstrates an advanced ad targeting system for streaming platforms that:
- Predicts CTR using Random Forest classification
- Delivers personalized ad content based on user profiles
- Features a Netflix-inspired UI with video playback
- Includes real-time analytics with speedometer-style dashboards

## Demo Features

### 🎬 Netflix-Style Interface
- Black-themed UI with red accents
- Video streaming with multiple shows
- Interactive movie selection
- Responsive design for all devices

### 📊 ML-Powered Ad Optimization
- **Algorithm**: Random Forest Classifier (100 trees, max depth 10)
- **Features**: User demographics, viewing habits, content context
- **Real-time CTR prediction**: 0.02-0.08 average
- **Personalized ad copy** using MockLLM

### 🎯 Ad Experience
- Video playback with ad insertion after 10 seconds
- Skip countdown timer (5 seconds)
- Click tracking and performance analytics
- A/B testing capabilities

### 💬 Interactive Chat Assistant
- Ask about campaign performance
- Get insights on CTR metrics
- Learn about optimization strategies

### 📈 Analytics Dashboard
- Speedometer-style gauges for real-time metrics
- Live performance charts
- Top performing campaigns
- Hourly tracking

## Tech Stack

- **Backend**: Flask 2.3.2 (Python)
- **ML Framework**: Scikit-learn 1.3.0
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js, GaugeJS
- **Video**: HTML5 Video Player

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**

'git clone https://github.com/cruizviquez/Micro-Netflix-Ads-Ctr.git
'cd Micro-Netflix-Ads-Ctr


 ### Run the setup script

'chmod +x run.sh
'./run.sh


### Start the application

'python3 app.py


### Access the application

    Open browser to http://localhost:5000
    In GitHub Codespaces, use the forwarded URL


## Project Structure


Micro-Netflix-Ads-Ctr/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── run.sh                # Startup script
├── templates/            # HTML templates
│   ├── index.html        # Netflix-style homepage
│   ├── dashboard.html    # Analytics dashboard
│   └── test.html         # Testing interface
├── static/
│   ├── css/
│   │   ├── netflix-style.css     # Main styles
│   │   └── dashboard-gauges.css  # Dashboard styles
│   └── js/
│       ├── netflix-app.js        # Main app logic
│       └── dashboard-gauges.js   # Dashboard scripts
└── data/                 # Generated test data



## Usage
### Viewing Experience

    Select a movie from the homepage
    Video starts playing automatically
    After 10 seconds, a personalized ad appears
    Skip or click the ad to continue

### Chat Assistant

    Click the chat widget (bottom-right)
    Ask questions like:
        "What's the current CTR?"
        "Show me performance metrics"
        "How does optimization work?"

### Analytics Dashboard

    Navigate to /dashboard
    View real-time metrics
    Monitor campaign performance
    Export reports


## API Endpoints
POST /api/optimize_ad       # Get optimized ad for user
POST /api/report_click      # Track ad interactions
POST /api/chat              # Chat assistant
GET  /api/test_llm          # Test LLM functionality
GET  /api/generate_test_data # Generate sample data
GET  /dashboard             # Analytics dashboard




## ML Model Details
Features Used

    User Profile: Age, gender, subscription tier, watch hours
    Content Context: Genre preference, time of day
    Ad Attributes: Category, creative freshness

## Performance Metrics

    Average CTR: 3.8%
    Top campaigns: 7.2% CTR
    Model accuracy: ~85%

## Sample Data

The system includes synthetic data generation for testing:

    5000 training samples
    User demographics distribution
    Realistic viewing patterns
    Click behavior simulation

## Contributing

Feel free to fork and submit pull requests. Areas for improvement:

    Real LLM integration (GPT-4, Claude)
    Advanced RL algorithms
    More sophisticated targeting
    Production-ready deployment

## Future Enhancements

Real-time bidding system
Multi-armed bandit optimization
Deep learning models
Actual video ad integration
Production database

 
## User authentication

License

MIT License - See LICENSE file for details
Author: Dr. Carlos Ruiz Viquez


  #  Video samples from Google's public dataset
    Inspired by Netflix's UI/UX
    Built for demonstrating ML capabilities in ad tech
