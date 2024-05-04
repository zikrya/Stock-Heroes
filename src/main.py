from api.alpha_vantage import fetch_continuous_updates

def main():
    api_key = 'GZY5P76URW9V54GH'
    symbol = 'IBM'
    interval = '5min'
    update_interval = 300

    print(fetch_continuous_updates(symbol, interval, api_key, update_interval))

if __name__ == "__main__":
    main()
