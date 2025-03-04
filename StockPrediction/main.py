from flask import Flask, request, jsonify, render_template
import keras
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET'])
def predict():

    end_date = request.args.get('date')
    model_name = request.args.get('model_name')

    # Choose a model
    if model_name == "LSTM":
        model = keras.models.load_model("lstm_model.keras")
        scaler = joblib.load('lstm_scaler.pkl')
        window_size = 5
    elif model_name == "RNN":
        model = keras.models.load_model("rnn_model.keras")
        scaler = joblib.load('rnn_scaler.pkl')
        window_size = 5
    elif model_name == "LSTM_CNN":
        model = keras.models.load_model("lstm_cnn_model.keras")
        scaler = joblib.load('lstm_cnn_scaler.pkl')
        window_size = 5
    elif model_name == "GRU_CNN":
        model = keras.models.load_model("gru_cnn_model.keras")
        scaler = joblib.load('gru_cnn_scaler.pkl')
        window_size = 10


    # judge whether the input is null
    if not end_date:
        return jsonify({"error": "Date not provided"}), 400

    # Get historical data
    start_date = datetime.strptime(end_date, "%Y-%m-%d")
    stock = yf.Ticker("^GSPC")
    current = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    # Judge whether the date is available
    data_end = stock.history(start=start_date, end=current)
    if data_end.empty:
        return jsonify({"error": "No available data"}), 401

    # Get enough historical data
    while True:

        start_date -= timedelta(days=window_size)
        data = stock.history(start=start_date.strftime("%Y-%m-%d"), end=datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1))
        if len(data) >= window_size:
            break

    # Record the data
    data = data.tail(window_size)
    dates = [d.strftime("%Y-%m-%d") for d in data.index]
    pred_date = data.index[-1] + timedelta(days=1)
    dates.append(pred_date.strftime("%Y-%m-%d"))
    data = data[["Close"]].to_numpy()

    # Predict the result
    X_input = scaler.transform(data).reshape(1, window_size, 1)
    y_pred = model.predict(X_input)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    data_list = [item[0] for item in data.tolist()]
    data_list.append(float(y_pred[0][0]))

    # Convert the format
    result = {
        "dates": dates,
        "close": data_list,
    }

    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=False)

