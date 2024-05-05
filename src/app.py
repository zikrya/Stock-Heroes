from flask import Flask, render_template, jsonify
import threading
import time
from api.polygon_api import fetch_stock_data

app = Flask(__name__)

symbols = ['AAPL']
latest_data = {}

@app.route('/api/data')
def api_data():
    return jsonify(latest_data)

def update_data(api_key, update_interval):
    while True:
        for symbol in symbols:
            data = fetch_stock_data(symbol, api_key)
            if isinstance(data, dict):
                latest_data[symbol] = data
            else:
                print("Error fetching data for {}: {}".format(symbol, data))
        time.sleep(update_interval)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
    update_interval = 300
    thread = threading.Thread(target=update_data, args=(api_key, update_interval))
    thread.daemon = True
    thread.start()
    app.run(debug=True)

