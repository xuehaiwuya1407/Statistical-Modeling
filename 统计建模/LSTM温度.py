# 文件1：model_full_2013_2023.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
df = pd.read_excel(r"C:\\Users\\dongx\\Desktop\\北京-逐日1.xlsx")
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)
data = df.loc['2013-12-02':'2023-02-15', ['平均气温', 'AQI']].copy()
data.fillna(method='ffill', inplace=True)

# 归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 构造序列
def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :-1])
        y.append(data[i+time_steps, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 训练集和测试集
split_date = '2021-01-01'
split_idx = (df.index.get_loc(split_date) - 7)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32)

# 模型
class HybridLSTMGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        return self.fc(gru_out[:, -1, :])

model = HybridLSTMGRU(input_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练
for epoch in range(500):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}")

# 预测与还原
model.eval()
preds, actual = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        out = model(batch_X.to(device)).cpu().numpy()
        preds.extend(out)
        actual.extend(batch_y.cpu().numpy())

preds, actual = np.array(preds), np.array(actual)

# 反归一化
dummy = np.zeros((len(preds), 1))
pred_real = scaler.inverse_transform(np.hstack((dummy, preds)))[:, 1]
y_real = scaler.inverse_transform(np.hstack((dummy, actual)))[:, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 误差图
errors = pred_real - y_real
plt.figure(figsize=(12,6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel("预测误差")
plt.ylabel("频数")
plt.title("2013-2023 模型预测误差分布")
plt.grid(True)
plt.show()
