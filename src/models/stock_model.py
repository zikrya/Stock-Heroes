import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/Users/zikryajaved/Desktop/Python Projects/stock-heroes/src')
from api.ml_data_handler import fetch_historical_data, normalize_data, create_sequences
from api.data_splitter import split_data
from models.model_evaluation import evaluate_model

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(StockPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



# Load and prepare data
api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2023-01-01'
raw_data = fetch_historical_data(symbol, api_key, start_date, end_date)
normalized_data = normalize_data(raw_data)
X, y = create_sequences(normalized_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y, train_size=0.8)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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

# Optionally evaluate on the testing set after training
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)
    print(f'Test Loss: {test_loss.item()}')
    # Convert predictions and actuals to numpy for evaluation
    test_preds_np = test_preds.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    evaluate_model(test_preds_np, y_test_np)  # Evaluate the model
