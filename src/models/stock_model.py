import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/Users/zikryajaved/Desktop/Python Projects/stock-heroes/src')
from api.ml_data_handler import fetch_historical_data, normalize_data, create_sequences

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        print(f"Initial hidden state shape: {h0.shape}")
        print(f"Initial cell state shape: {c0.shape}")
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load and prepare data
api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
symbol = 'IBM'
start_date = '2021-01-01'
end_date = '2023-01-01'
raw_data = fetch_historical_data(symbol, api_key, start_date, end_date)
normalized_data = normalize_data(raw_data)
X, y = create_sequences(normalized_data)

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print(f"Shape of X_train: {X_train.shape}")  # Check input shape
print(f"Shape of y_train: {y_train.shape}")  # Check output shape

# Instantiate the model
model = StockPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
