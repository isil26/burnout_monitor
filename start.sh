#!/bin/bash

# 🚀 Enron Organizational Dynamics Monitor - Quick Start Script
# Automated setup and launch for portfolio demonstration

echo "📊 Enron Organizational Dynamics Monitor - Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if emails.csv exists
if [ ! -f "emails.csv" ]; then
    echo "❌ ERROR: emails.csv not found in current directory"
    echo ""
    echo "Please download the Enron Email Dataset and place it here as 'emails.csv'"
    echo "Dataset source: https://www.cs.cmu.edu/~enron/"
    echo ""
    exit 1
fi

echo "✅ Dataset found: emails.csv"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Download TextBlob corpora
echo "📚 Downloading NLP corpora..."
python -m textblob.download_corpora lite > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ NLP corpora downloaded"
else
    echo "⚠️  Warning: Could not download NLP corpora (may already exist)"
fi
echo ""

# Display information
echo "=================================================="
echo "🎉 Setup Complete!"
echo "=================================================="
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo ""
echo "⚙️  Quick Settings:"
echo "   • Sample Mode: 2,000 emails (default, ~15 seconds)"
echo "   • Full Dataset: 500K+ emails (~60 minutes)"
echo ""
echo "🎯 Demo Tips:"
echo "   1. Start with sample mode for quick iteration"
echo "   2. Use filters to explore different scenarios"
echo "   3. Hover over network nodes for detailed metrics"
echo ""
echo "=================================================="
echo ""
echo "🚀 Launching Streamlit application..."
echo ""

# Launch Streamlit
streamlit run app.py

# Deactivate virtual environment on exit
deactivate
