from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Advanced Sentiment Analysis Function
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]  # Analyze sentiment using the pre-trained model
    label = result['label'].lower()

    # Map emotions to depression-specific states
    depression_states = {
        "sadness": "depressed",
        "fear": "anxious",
        "anger": "hopeless",
        "love": "encouraged",
        "joy": "encouraged",
        "surprise": "lonely",
        "neutral": "neutral"
    }
    return depression_states.get(label, "neutral")


# Function to get recommendations based on sentiment
def get_recommendations(sentiment):
    data = pd.read_csv("resources.csv")
    recommendations = data[data['sentiment'] == sentiment]
    return recommendations.sample(3).to_dict(orient="records")

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = analyze_sentiment(text)
    recommendations = get_recommendations(sentiment)
    return render_template("results.html", sentiment=sentiment, recommendations=recommendations)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
