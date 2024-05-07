import requests
import pandas as pd
import numpy as np

def fetch_historical_data(symbol, api_key, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=120&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        if 'results' in json_data:
            df = pd.DataFrame(json_data['results'])
            print(df.columns)
            return df
        else:
            print(f"No data available for {symbol}")
            return pd.DataFrame()
    else:
        print(f"Failed to fetch data for {symbol}: Status Code {response.status_code}")
        return pd.DataFrame()

def normalize_data(df):
    df = df[['o', 'h', 'l', 'c', 'v']]
    df = (df - df.min()) / (df.max() - df.min())
    return df

def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i-n_steps:i].values)
        y.append(data.iloc[i]['c'])
    return np.array(X), np.array(y)