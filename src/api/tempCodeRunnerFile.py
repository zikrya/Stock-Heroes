from polygon import RESTClient
import time

from polygon import RESTClient

def fetch_stock_data(symbol, api_key):
    client = RESTClient(api_key)
    try:
        resp = client.get_last_trade(symbol)
        if resp:
            return {'price': resp.price, 'timestamp': resp.timestamp}
        else:
            print(f"No data available for {symbol}")
            return None
    except Exception as e:
        print(f"An error occurred fetching data for {symbol}: {e}")
        return None
    finally:
        client.close()


def fetch_continuous_updates(symbols, api_key, update_interval):
    while True:
        for symbol in symbols:
            data = fetch_stock_data(symbol, api_key)
            if isinstance(data, dict):
                print(f"{symbol}: {data}")
            else:
                print(f"Error for {symbol}: {data}")
        time.sleep(update_interval)
