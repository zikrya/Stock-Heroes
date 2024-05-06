import requests
import pandas as pd
import numpy as np

def fetch_historical_data(symbol, api_key, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=120&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            if 'results' in json_data and json_data['results']:
                result = json_data['results']
                df = pd.DataFrame(result)
                print(df.columns)  # Debug: Print columns to verify
                return df
            else:
                print(f"No data available for {symbol}")
                return pd.DataFrame()
        else:
            print(f"Failed to fetch data for {symbol}: Status Code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def normalize_data(df):
    if 'o' in df.columns and 'h' in df.columns and 'l' in df.columns and 'c' in df.columns and 'v' in df.columns:
        df = df[['o', 'h', 'l', 'c', 'v']]  # Select only the required columns
        for column in df.columns:
            max_value = df[column].max()
            min_value = df[column].min()
            df[column] = (df[column] - min_value) / (max_value - min_value)
    return df

def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i-n_steps:i].values)
        y.append(data.iloc[i]['c'])  # Assuming 'c' is the close price
    return np.array(X), np.array(y)

api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2023-01-01'
data = fetch_historical_data(symbol, api_key, start_date, end_date)
data = normalize_data(data)
X, y = create_sequences(data)
print(X, y)
