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
        self.attention = Attention(hidden_dim * 2, 60)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
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