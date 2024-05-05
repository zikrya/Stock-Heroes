import requests
import time

def fetch_stock_data(symbol, api_key):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-01-09/2023-01-09?adjusted=true&sort=asc&limit=120&apiKey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                # Convert the timestamp from milliseconds to a readable date-time format
                timestamp = data['results'][0]['t']
                readable_date = datetime.utcfromtimestamp(timestamp / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                return {'price': data['results'][0]['c'], 'timestamp': readable_date}
            else:
                print(f"No data available for {symbol}")
                return None
        else:
            print(f"Failed to fetch data for {symbol}: Status Code {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def fetch_continuous_updates(symbols, api_key, update_interval):
    while True:
        for symbol in symbols:
            data = fetch_stock_data(symbol, api_key)
            if isinstance(data, dict):
                print(f"{symbol}: {data}")
            else:
                print(f"Error for {symbol}: {data}")
        time.sleep(update_interval)