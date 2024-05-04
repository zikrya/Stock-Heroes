from flask import Flask, render_template, jsonify
import threading
from api.alpha_vantage import fetch_continuous_updates

app = Flask(__name__)

# Setting up a global dictionary to hold the latest data
latest_data = {}

def update_data(symbol, interval, api_key, update_interval):
    global latest_data
    while True:
        latest_data[symbol] = fetch_continuous_updates(symbol, interval, api_key, update_interval)

@app.route('/')
def home():
    # Assuming you want to display data for 'IBM'
    symbol = 'IBM'
    if symbol in latest_data:
        return render_template('index.html', stock_data=latest_data[symbol])
    else:
        return "Waiting for data..."

if __name__ == "__main__":
    api_key = 'GZY5P76URW9V54GH'
    symbol = 'IBM'
    interval = '5min'
    update_interval = 300

    # Start a background thread to update stock data continuously
    thread = threading.Thread(target=update_data, args=(symbol, interval, api_key, update_interval))
    thread.daemon = True
    thread.start()

    app.run(debug=True)
