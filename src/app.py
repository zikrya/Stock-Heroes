from flask import Flask, render_template
import threading
import time
from api.alpha_vantage import fetch_stock_data, fetch_continuous_updates

app = Flask(__name__)

# This will hold the latest fetched data
latest_data = {}

def update_data(symbol, interval, api_key, update_interval):
    global latest_data
    # This loop will continuously update the latest data
    while True:
        data = fetch_stock_data(symbol, interval, api_key)
        if isinstance(data, dict):
            latest_data[symbol] = data  # Store the latest data for the symbol
        time.sleep(update_interval)

@app.route('/')
def home():
    symbol = 'IBM'
    data = latest_data.get(symbol, None)
    if data:
        # Extracting the latest timestamp's data
        latest_point = next(iter(data['Time Series (5min)'].values()))
        return render_template('index.html', stock_data=latest_point)
    else:
        return render_template('index.html', stock_data=None)

if __name__ == '__main__':
    api_key = 'GZY5P76URW9V54GH'
    symbol = 'IBM'
    interval = '5min'
    update_interval = 300  # Update every 5 minutes

    # Starting a background thread to fetch data continuously
    thread = threading.Thread(target=update_data, args=(symbol, interval, api_key, update_interval))
    thread.daemon = True
    thread.start()

    app.run(debug=True)
