from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

app = Flask(__name__)

# Load the IMDB dataset and word index
max_features = 10000
maxlen = 200
(x_train, _), (_, _) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()

# Load LSTM and GRU models
lstm_model = load_model('model_lstm_final.h5')
#gru_model = load_model('gru_model.h5')

# Function to preprocess text and predict sentiment
def predict_sentiment(model, text):
    words = text.split()
    words = [word_index[word] + 3 for word in words if word in word_index]
    words = pad_sequences([words], maxlen=maxlen)
    prediction = model.predict(words)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return prediction, sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model']
    text = request.form['text']
    if model_type == 'lstm':
        prediction, sentiment = predict_sentiment(lstm_model, text)
    # elif model_type == 'gru':
    #     prediction, sentiment = predict_sentiment(gru_model, text)
    else:
        return "Invalid model selection!"
    
    if sentiment == "Positive":
        image = "happy.jpg"
    else:
        image = "sad.jpg"
    
    return render_template('result.html', prediction=prediction, sentiment=sentiment, image=image)

if __name__ == '__main__':
    app.run(debug=True)
