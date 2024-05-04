import requests
import time

def fetch_stock_data(symbol, interval, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Failed to fetch data: Status Code {response.status_code}"

def fetch_continuous_updates(symbol, interval, api_key, update_interval):
    while True:
        data = fetch_stock_data(symbol, interval, api_key)
        if isinstance(data, dict):
            print(data)
        else:
            print(data)
        time.sleep(update_interval)
