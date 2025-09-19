#!/bin/bash

echo "🎬 Starting Microflix Ad Demo..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p templates static/css static/js data

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting Flask server..."
echo ""

# Run the app
python app.py