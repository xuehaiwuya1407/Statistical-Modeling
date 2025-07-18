import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 数据加载与预处理
df = pd.read_excel(r"C:\Users\dongx\Desktop\北京-逐日1.xlsx")
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)

input_features = ['平均气温', '平均湿度']
target_col = 'AQI'

# 2. 拆分数据（2013-2020 和 2021-2023）
data_2013_2020 = df.loc[:'2020-12-31', input_features + [target_col]].copy()
data_2021_2023 = df.loc['2021-01-01':, ['平均气温', target_col]].copy()  # 只使用温度

# 填充缺失值
data_2013_2020.fillna(method='ffill', inplace=True)
data_2021_2023.fillna(method='ffill', inplace=True)

# 归一化
scaler_2013_2020 = MinMaxScaler()
scaler_2021_2023 = MinMaxScaler()

scaled_data_2013_2020 = scaler_2013_2020.fit_transform(data_2013_2020)
scaled_data_2021_2023 = scaler_2021_2023.fit_transform(data_2021_2023)

# 3. 滑动窗口构造序列
def create_sequences(data, time_steps=7, target_index=-1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :-1])  # 输入特征
        y.append(data[i+time_steps, target_index])  # 目标值
    return np.array(X), np.array(y)

target_index = scaled_data_2013_2020.shape[1] - 1  # AQI 的列索引

# 为 2013-2020 数据构造序列
X_2013_2020, y_2013_2020 = create_sequences(scaled_data_2013_2020, time_steps=7, target_index=target_index)

# 为 2021-2023 数据构造序列（不使用湿度）
target_index_2021_2023 = scaled_data_2021_2023.shape[1] - 1
X_2021_2023, y_2021_2023 = create_sequences(scaled_data_2021_2023, time_steps=7, target_index=target_index_2021_2023)

# 划分训练集和验证集（80% 训练，20% 验证）
split_2013_2020 = int(len(X_2013_2020) * 0.8)
split_2021_2023 = int(len(X_2021_2023) * 0.8)

X_train_2013_2020, y_train_2013_2020 = X_2013_2020[:split_2013_2020], y_2013_2020[:split_2013_2020]
X_val_2013_2020, y_val_2013_2020 = X_2013_2020[split_2013_2020:], y_2013_2020[split_2013_2020:]

X_train_2021_2023, y_train_2021_2023 = X_2021_2023[:split_2021_2023], y_2021_2023[:split_2021_2023]
X_val_2021_2023, y_val_2021_2023 = X_2021_2023[split_2021_2023:], y_2021_2023[split_2021_2023:]

# 转换为 torch 张量
X_train_2013_2020 = torch.tensor(X_train_2013_2020, dtype=torch.float32)
y_train_2013_2020 = torch.tensor(y_train_2013_2020, dtype=torch.float32).view(-1, 1)
X_val_2013_2020 = torch.tensor(X_val_2013_2020, dtype=torch.float32)
y_val_2013_2020 = torch.tensor(y_val_2013_2020, dtype=torch.float32).view(-1, 1)

X_train_2021_2023 = torch.tensor(X_train_2021_2023, dtype=torch.float32)
y_train_2021_2023 = torch.tensor(y_train_2021_2023, dtype=torch.float32).view(-1, 1)
X_val_2021_2023 = torch.tensor(X_val_2021_2023, dtype=torch.float32)
y_val_2021_2023 = torch.tensor(y_val_2021_2023, dtype=torch.float32).view(-1, 1)

# 4. 定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 5. 定义模型（LSTM + GRU）
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
        out = gru_out[:, -1, :]
        return self.fc(out)

# 6. 训练和验证函数
def train_and_evaluate(model, train_loader, val_loader, epochs=200, patience=50):
    best_val_loss = float('inf')
    trigger_times = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model

# 7. 训练两个模型（2013-2020 和 2021-2023）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2013-2020模型
train_loader_2013_2020 = DataLoader(TimeSeriesDataset(X_train_2013_2020, y_train_2013_2020), batch_size=32, shuffle=True)
val_loader_2013_2020 = DataLoader(TimeSeriesDataset(X_val_2013_2020, y_val_2013_2020), batch_size=32)
model_2013_2020 = HybridLSTMGRUModel(input_size=2).to(device)  # 使用两个特征
model_2013_2020 = train_and_evaluate(model_2013_2020, train_loader_2013_2020, val_loader_2013_2020)

# 2021-2023模型
train_loader_2021_2023 = DataLoader(TimeSeriesDataset(X_train_2021_2023, y_train_2021_2023), batch_size=32, shuffle=True)
val_loader_2021_2023 = DataLoader(TimeSeriesDataset(X_val_2021_2023, y_val_2021_2023), batch_size=32)
model_2021_2023 = HybridLSTMGRUModel(input_size=1).to(device)  # 只有温度作为输入
model_2021_2023 = train_and_evaluate(model_2021_2023, train_loader_2021_2023, val_loader_2021_2023)

# 8. 预测与可视化
# 2013-2020预测
model_2013_2020.eval()
with torch.no_grad():
    predictions_2013_2020 = model_2013_2020(X_val_2013_2020.to(device)).cpu().numpy()
    y_val_2013_2020_np = y_val_2013_2020.cpu().numpy()

# 2021-2023预测
model_2021_2023.eval()
with torch.no_grad():
    predictions_2021_2023 = model_2021_2023(X_val_2021_2023.to(device)).cpu().numpy()
    y_val_2021_2023_np = y_val_2021_2023.cpu().numpy()

# 反归一化：为了还原 AQI 数值，我们构造一个 dummy 特征矩阵
def inverse_transform_AQI(aqi_values, scaler, feature_num):
    dummy = np.zeros((aqi_values.shape[0], feature_num))
    combined = np.hstack((dummy, aqi_values))
    inv = scaler.inverse_transform(combined)
    return inv[:, -1]

# 2013-2020还原（使用两个特征时的 scaler）
y_pred_2013_2020_real = inverse_transform_AQI(predictions_2013_2020, scaler_2013_2020, feature_num=len(input_features))
y_val_2013_2020_real = inverse_transform_AQI(y_val_2013_2020_np, scaler_2013_2020, feature_num=len(input_features))

# 2021-2023还原（使用1个输入特征）
y_pred_2021_2023_real = inverse_transform_AQI(predictions_2021_2023, scaler_2021_2023, feature_num=1)
y_val_2021_2023_real = inverse_transform_AQI(y_val_2021_2023_np, scaler_2021_2023, feature_num=1)

# 计算指标
mse = mean_squared_error(y_val_2013_2020_real, y_pred_2013_2020_real)
mae = mean_absolute_error(y_val_2013_2020_real, y_pred_2013_2020_real)
print(f'2013-2020 模型 => MSE: {mse:.4f}, MAE: {mae:.4f}')

# -----------------------------
# 数据可视化部分
# -----------------------------
# 可视化 2013-2020 模型预测结果
plt.figure(figsize=(12,6))
plt.plot(y_val_2013_2020_real, label="实际 AQI", marker='o', linestyle='-')
plt.plot(y_pred_2013_2020_real, label="预测 AQI", marker='x', linestyle='--')
plt.xlabel("样本序号")
plt.ylabel("AQI")
plt.title("2013-2020 实际值 vs. 预测值")
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测误差分布图（残差图）
errors = y_pred_2013_2020_real - y_val_2013_2020_real
plt.figure(figsize=(12,6))
sns.histplot(errors, kde=True, bins=30)
plt.xlabel("预测误差 (Predicted - Actual)")
plt.ylabel("频数")
plt.title("2013-2020 模型预测误差分布")
plt.show()

# 可视化 2021-2023 模型预测结果（示例）
plt.figure(figsize=(12,6))
plt.plot(y_val_2021_2023_real, label="实际 AQI", marker='o', linestyle='-')
plt.plot(y_pred_2021_2023_real, label="预测 AQI", marker='x', linestyle='--')
plt.xlabel("样本序号")
plt.ylabel("AQI")
plt.title("2021-2023 实际值 vs. 预测值")
plt.legend()
plt.grid(True)
plt.show()

# 绘制 2021-2023 模型预测误差分布图
errors_2021_2023 = y_pred_2021_2023_real - y_val_2021_2023_real
plt.figure(figsize=(12,6))
sns.histplot(errors_2021_2023, kde=True, bins=30)
plt.xlabel("预测误差 (Predicted - Actual)")
plt.ylabel("频数")
plt.title("2021-2023 模型预测误差分布")
plt.show()