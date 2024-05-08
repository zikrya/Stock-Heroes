import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/Users/zikryajaved/Desktop/Python Projects/stock-heroes/src')
from api.ml_data_handler import fetch_historical_data, normalize_data, create_sequences
from api.data_splitter import split_data
from models.model_evaluation import evaluate_model

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.weights = nn.Parameter(torch.Tensor(feature_dim, 1))
        nn.init.kaiming_uniform_(self.weights)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.bias, 0.1)

    def forward(self, x):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.weights) + self.bias
        eij = eij.view(-1, self.step_dim)
        a = torch.softmax(eij, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(StockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim * 2, 60)  # Assuming sequence length is 60
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)  # Adding batch normalization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.attention(out)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out



def load_model(path='model_weights.pth'):
    model = StockPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_and_interpret(model, data):
    X, _ = create_sequences(data)
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X)
    predicted_prices = predictions.numpy()
    advice = 'Buy' if predicted_prices[-1] > predicted_prices[-2] else 'Sell'
    return predicted_prices, advice
# Load and prepare data
api_key = 'aPGsprF96a0EzYQeDq8Ypgjkr1MGRxsM'
symbol = 'IBM'
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