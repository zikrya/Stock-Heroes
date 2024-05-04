import requests

def fetch_stock_data(info):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=GZY5P76URW9V54GH'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Failed to fetch data: Status Code {response.status_code}"
