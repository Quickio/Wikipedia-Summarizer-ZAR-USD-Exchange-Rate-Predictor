import wikipedia
from transformers import pipeline
from fpdf import FPDF

name = input("What is your name")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def fetch_wikipedia_summary(topic):
    try:
       
        wiki_content = wikipedia.page(topic).content
        print("✅ Fetched Wikipedia content.")
        return wiki_content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Topic too broad. Try one of: {e.options[:5]}")
    except wikipedia.exceptions.PageError:
        print("❌ Topic not found.")
    return None

def summarize_text(text, max_words=1000):
   
    chunks = []
    while len(text) > 1000:
        split_idx = text[:1000].rfind('.')
        chunks.append(text[:split_idx+1])
        text = text[split_idx+1:]
    chunks.append(text)

    summary = ""
    for chunk in chunks:
        summary += summarizer(chunk, max_length=150, min_length=60, do_sample=False)[0]['summary_text'] + " "
    print("✅ Summarized content.")
    return summary.strip()

def create_pdf(topic, content, filename="Topic_Summary.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=topic.upper(), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line)
    
    pdf.output(filename)
    print(f"✅ your pdf is now ready {name} : {filename}")


if __name__ == "__main__":
    user_topic = input("🔍 Enter a topic: ")
    raw_text = fetch_wikipedia_summary(user_topic)
    if raw_text:
        summary = summarize_text(raw_text[:5000]) 
        create_pdf(user_topic, summary, filename=f"{user_topic}_Summary.pdf")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def get_data():
   
    data = yf.download('ZAR=X', start='2010-01-01', end='2025-01-01')
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

   
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_and_predict(data):
    look_back = 60
    X, y, scaler = prepare_data(data, look_back)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return y_test, predictions, model, scaler, X_test



def plot_results(y_test, predictions):
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='blue', label='Actual ZAR/USD')
    plt.plot(predictions, color='red', label='Predicted ZAR/USD')
    plt.title('ZAR/USD Exchange Rate Prediction')
    plt.xlabel('Time')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = get_data()
    y_test, predictions, model, scaler, X_test = train_and_predict(data)
    plot_results(y_test, predictions)
