from flask import Flask, render_template, request, jsonify
import torch
from models.stock_model import StockPredictor, load_model
from api.ml_data_handler import fetch_historical_data, normalize_data, create_sequences

app = Flask(__name__)
model = load_model('/Users/zikryajaved/Desktop/Python Projects/stock-heroes/src/models/model_weights.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Change to handle JSON data
    symbol = data['symbol']
    raw_data = fetch_historical_data(symbol, 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM', '2021-01-01', '2023-01-01')
    if raw_data.empty:
        return jsonify({'error': 'No data available for this symbol'})

    normalized_data = normalize_data(raw_data)
    X, _ = create_sequences(normalized_data)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(X_tensor)
        advice = 'Buy' if predictions[-1] > predictions[-2] else 'Sell'

    return jsonify({'predictions': predictions.tolist(), 'advice': advice})

if __name__ == '__main__':
    app.run(debug=True)

