from flask import Flask, render_template, jsonify
import threading
import time
from api.alpha_vantage import fetch_stock_data

app = Flask(__name__)

# This will hold the latest fetched data
latest_data = {}

def update_data(symbol, interval, api_key, update_interval):
    global latest_data
    while True:
        data = fetch_stock_data(symbol, interval, api_key)
        if isinstance(data, dict):
            latest_data[symbol] = data  # Store the latest data for the symbol
        else:
            print(data)  # Log error message
        time.sleep(update_interval)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    symbol = 'IBM'
    data = latest_data.get(symbol, None)
    if data:
        # Extracting the latest timestamp's data
        latest_point = next(iter(data['Time Series (5min)'].values()))
        return jsonify(latest_point)
    else:
        return jsonify({'error': 'No data available'})

if __name__ == '__main__':
    api_key = 'YOUR_API_KEY'  # Replace with your actual API key
    symbol = 'IBM'
    interval = '5min'
    update_interval = 300  # Update every 5 minutes

    # Starting a background thread to fetch data continuously
    thread = threading.Thread(target=update_data, args=(symbol, interval, api_key, update_interval))
    thread.daemon = True
    thread.start()

    app.run(debug=True)

