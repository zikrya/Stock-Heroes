import requests

def fetch_stock_data(symbol):
    url = f"https://api.iextrading.com/1.0/stock/{symbol}/quote"
    response = requests.get(url)
    data = response.json()
    return data

# Test the function with a stock symbol, e.g., 'AAPL' for Apple Inc.
if __name__ == "__main__":
    stock_data = fetch_stock_data('AAPL')
    print(stock_data)
