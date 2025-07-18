# model_short_2013_2020.py
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

# 1. 数据加载
file_path = r"C:\\Users\\dongx\\Desktop\\北京-逐日1.xlsx"
df = pd.read_excel(file_path)
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 2. 数据选择
full_data = df.loc['2013-12-02':'2020-12-31', ['AQI', '平均气温', '平均湿度']].copy()
test_data = df.loc['2021-01-01':'2023-02-15', ['AQI', '平均气温', '平均湿度']].copy()

full_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

# 3. 归一化
scaler = MinMaxScaler()
scaled_full = scaler.fit_transform(full_data)
scaled_test = scaler.transform(test_data)

# 4. 构造序列
def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 1:])  # 平均气温与湿度
        y.append(data[i+time_steps, 0])     # AQI
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(scaled_full)
X_test, y_test = create_sequences(scaled_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 5. 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 6. 模型定义
class HybridLSTMGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        return self.fc(gru_out[:, -1, :])

# 7. 训练函数
def train(model, train_loader, val_loader, epochs=200, patience=50):
    best_loss = float('inf')
    trigger_times = 0
    best_model = None  # 初始化best_model为None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()  # 只有在更新best_loss时保存模型
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    if best_model is not None:  # 确保best_model被赋值后再加载
        model.load_state_dict(best_model)
    return model


# 8. 训练与测试
batch_size = 32
train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridLSTMGRUModel(input_size=2).to(device)
model = train(model, train_loader, test_loader)

# 9. 预测与评估
model.eval()
preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch).cpu().numpy()
        preds.append(output)
preds = np.vstack(preds)

# 反归一化
def inverse_aqi(pred, scaler):
    dummy = np.zeros((pred.shape[0], 3))
    dummy[:, 0] = pred[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]

preds_inv = inverse_aqi(preds, scaler)
y_test_inv = inverse_aqi(y_test.numpy(), scaler)

mse = mean_squared_error(y_test_inv, preds_inv)
mae = mean_absolute_error(y_test_inv, preds_inv)
print(f"模型 MSE: {mse:.2f}, MAE: {mae:.2f}")

# 误差分布图
errors = preds_inv - y_test_inv
plt.figure(figsize=(10,5))
sns.histplot(errors, bins=30, kde=True)
plt.title("2021-2023 测试集误差分布")
plt.xlabel("预测误差")
plt.ylabel("频数")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="实际 AQI", marker='o', linestyle='-')
plt.plot(preds_inv, label="预测 AQI", marker='x', linestyle='--')
plt.xlabel("样本序号")
plt.ylabel("AQI")
plt.title("2021-2023 实际值 vs. 预测值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
