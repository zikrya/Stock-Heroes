import torch
import torch.nn as nn
import torch.optim as optim
from stock_model import StockPredictor
import sys
sys.path.append('/Users/zikryajaved/Desktop/Python Projects/stock-heroes/src')
from api.ml_data_handler import fetch_historical_data, normalize_data, create_sequences
from api.data_splitter import split_data

def train_model(api_key, symbol, start_date, end_date):
    # Load and prepare data
    data = fetch_historical_data(symbol, api_key, start_date, end_date)
    normalized_data = normalize_data(data)
    X, y = create_sequences(normalized_data)
    X_train, X_test, y_train, y_test = split_data(X, y, train_size=0.8)

    # Define the model, loss, and optimizer
    model = StockPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Training loop
    for epoch in range(50):  # number of epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the trained model weights
    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
    symbol = 'IBM'
    start_date = '2021-01-01'
    end_date = '2023-01-01'
    train_model(api_key, symbol, start_date, end_date)
