import requests

def fetch_stock_data(symbol):
    api_key = 'aAJV0txwcNGrYn1PYHWaj4kZAOoop9oV'
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Failed to fetch data: Status Code {response.status_code}"

if __name__ == "__main__":
    symbol = 'AAPL'
    stock_data = fetch_stock_data(symbol)
    print(stock_data)
