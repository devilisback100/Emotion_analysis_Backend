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
required_nltk_data = ['corpora/stopwords.zip',
                      'corpora/wordnet.zip', 'tokenizers/punkt.zip']

for data in required_nltk_data:
    try:
        nltk.data.find(data)
    except LookupError:
        nltk.download(data.split('/')[-1].split('.')
                      [0], download_dir='./nltk_data')

# Load pre-trained models and vectorizer
vectorizer = joblib.load('text_data_vectorizer.joblib')
Model = joblib.load('emotion_model.joblib')

# Initialize NLP tools
Lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub('[^A-Za-z ]+', ' ', text)
    return text


def tokenize_text(text):
    text = word_tokenize(text)
    return text


def stop_word_removal(text):
    text = [word for word in text if word not in stop_words]
    return text


def lemmatize_text(text):
    text = [Lemmatizer.lemmatize(word) for word in text]
    return text


def to_string(text):
    text = ' '.join(text)
    return text


def feature_text(text):
    return vectorizer.transform([text])


def text_helper(text):
    text = clean_text(text)
    text = tokenize_text(text)
    text = stop_word_removal(text)
    text = lemmatize_text(text)
    text = to_string(text)
    return feature_text(text)


def emotion_classification(user_input, Emotions):
    processed_input = text_helper(user_input)
    predicted_emotion_index = list(Model.predict(processed_input))
    return Emotions[str(predicted_emotion_index[0])]


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
    app.run(debug=True)
