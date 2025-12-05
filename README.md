# Enron Organizational Dynamics Monitor

This project analyzes the Enron email corpus to study organizational communication and sentiment. It uses Python for data parsing, network analysis, and sentiment classification.

## Main Features
- Parses raw emails to extract sender, receiver, date, subject, and body
- Sentiment analysis using TextBlob and VADER
- Constructs directed communication graphs with NetworkX
- Calculates centrality and community metrics
- Anonymizes personal data for privacy
- Visualizes results with Streamlit and Plotly

## Installation
1. Clone the repository
2. Install dependencies:
   ```zsh
   pip install -r requirements.txt
   python -m textblob.download_corpora
   ```
3. Add the Enron dataset as `emails.csv` in the project folder

## Usage
Run the dashboard:
```zsh
streamlit run app.py
```

## Limitations
- Sentiment accuracy limited by lexicon-based models
- Community detection uses a greedy algorithm
- Full dataset requires >8GB RAM
- No parallel processing

## Future Work
- Integrate transformer models for sentiment
- Support incremental/out-of-memory processing
- Extend parser for MIME/multipart formats
- Add temporal network analysis
- Statistical testing for sentiment trends

## Author
isil26 (isilozyigit@gmail.com)
