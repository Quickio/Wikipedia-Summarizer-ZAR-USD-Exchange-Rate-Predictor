# Wikipedia-Summarizer-ZAR-USD-Exchange-Rate-Predictor
Overview
This project has two main parts:

Wikipedia Summarizer:

Fetches the content of a Wikipedia page for a user-provided topic.

Summarizes the content using a transformer-based summarization model.

Saves the summary as a basic PDF.

ZAR/USD Exchange Rate Predictor:

Downloads historical exchange rate data (South African Rand to USD).

Prepares the data for time series prediction using LSTM (Long Short-Term Memory) neural networks.

Trains an LSTM model to predict future exchange rates.

Plots actual vs predicted exchange rates.


How to Use
Run the script.

When prompted, enter your name (for personalized PDF message).

Enter a topic to summarize from Wikipedia.

The program fetches, summarizes, and saves the summary as a PDF file named <topic>_Summary.pdf.

The program then fetches ZAR/USD data, trains the model, and displays a plot comparing actual and predicted exchange rates.

Dependencies
Make sure you have these Python packages installed:

wikipedia

transformers

fpdf

yfinance

pandas

numpy

matplotlib

scikit-learn

tensorflow

pip install wikipedia transformers fpdf yfinance pandas numpy matplotlib scikit-learn tensorflow

