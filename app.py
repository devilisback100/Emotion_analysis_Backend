from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import joblib
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is available
nltk.data.path.append('./nltk_data')
required_nltk_data = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']

for data in required_nltk_data:
    try:
        nltk.data.find(f'corpora/{data}')
    except LookupError:
        nltk.download(data, download_dir='./nltk_data')

# Load pre-trained models and vectorizer
vectorizer = joblib.load('text_data_vectorizer.joblib')
Model = joblib.load('emotion_model.joblib')

# Initialize NLP tools
Lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    return re.sub('[^A-Za-z ]+', ' ', text.lower())


def tokenize_text(text):
    return word_tokenize(text)


def stop_word_removal(text):
    return [word for word in text if word not in stop_words]


def lemmatize_text(text):
    return [Lemmatizer.lemmatize(word) for word in text]


def text_helper(text):
    text = clean_text(text)
    text = tokenize_text(text)
    text = stop_word_removal(text)
    text = lemmatize_text(text)
    return vectorizer.transform([' '.join(text)])


def emotion_classification(user_input, Emotions):
    processed_input = text_helper(user_input)
    predicted_emotion_index = Model.predict(processed_input)[0]
    return Emotions[str(predicted_emotion_index)]


app = Flask(__name__)
CORS(app)


@app.route('/classify_emotion', methods=['POST'])
def classify_emotion():
    data = request.get_json()
    user_input = data.get('text')
    Emotions = {
        "0": "sadness",
        "1": "joy",
        "2": "love",
        "3": "anger",
        "4": "fear",
        "5": "surprise"
    }
    result = emotion_classification(user_input, Emotions)
    return jsonify({"emotion": result})


if __name__ == '__main__':
    from gunicorn.app.wsgiapp import run
    run()
