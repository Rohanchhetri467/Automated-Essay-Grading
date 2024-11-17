from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import site
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from keras.models import load_model
import keras.backend as K

# Initialize Flask app
app = Flask(__name__, template_folder='templates')  # Specify template folder
CORS(app)  # Enable CORS for all routes

def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = [w for w in x.split() if w not in stop_words]
    return filtered_sentence

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[word])
    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)
    return vec

def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for i, essay in enumerate(essays):
        essay_vecs[i] = makeVec(essay, model, num_features)
    return essay_vecs

def convertToVec(text):
    if len(text) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
        clean_test_essays = [sent2word(text)]
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))
    return "0"

@app.route('/', methods=['GET'])
def index():
    # Render the index.html file located in the 'templates' directory
    return render_template('index.html')

@app.route('/', methods=['POST'])
def create_task():
    K.clear_session()
    data = request.get_json()
    app.logger.info(f"Received data: {data}")  # Debug log to see what is received
    if not data or "text" not in data:
        return jsonify({"error": "Invalid input"}), 400
    final_text = data["text"]
    score = convertToVec(final_text)
    K.clear_session()
    return jsonify({'score': score}), 201

if __name__ == '__main__':
    app.run(debug=True)
