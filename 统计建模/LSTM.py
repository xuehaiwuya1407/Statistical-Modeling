import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------- 数据加载与预处理 ----------
df = pd.read_excel(r"C:\Users\dongx\Desktop\北京-逐日1.xlsx")

# 空气质量等级映射
quality_map = {'优': 0, '良': 1, '轻度污染': 2, '中度污染': 3, '重度污染': 4, '严重污染': 5}
df['质量等级'] = df['质量等级'].map(quality_map)

# 时间设置为索引
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 选取特征，移除 '质量等级'
features = ['AQI', 'PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']
data = df[features].copy()

# 填充缺失值（如果有）
data.fillna(method='ffill', inplace=True)

# 归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ---------- 构造时序数据 ----------
def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps][0])  # AQI
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# 划分训练/测试集
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 转为 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ---------- 自定义Dataset ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---------- 定义LSTM模型 ----------
class LSTMAQIModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMAQIModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        return self.fc(out)

# ---------- 将模型和数据移动到GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMAQIModel(input_size=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------- 模型训练 ----------
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # 将数据移动到GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# ---------- 模型预测 ----------
model.eval()
with torch.no_grad():
    # 将数据移动到GPU
    X_test = X_test.to(device)
    predictions = model(X_test).cpu().numpy()  # 预测结果移动回CPU
    y_test_np = y_test.cpu().numpy()

# 反归一化
y_pred_real = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), len(features)-1)))))[:, 0]
y_test_real = scaler.inverse_transform(np.hstack((y_test_np, np.zeros((len(y_test_np), len(features)-1)))))[:, 0]

# ---------- 可视化 ----------
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label='真实值')
plt.plot(y_pred_real, label='预测值')
plt.title('PyTorch LSTM AQI预测效果')
plt.xlabel('时间步')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 计算特征重要性 ----------
# 使用敏感性分析方法计算特征重要性
def sensitivity_analysis(model, X_test, feature_index):
    model.eval()
    with torch.no_grad():
        original_predictions = model(X_test).cpu().numpy()
        X_test_perturbed = X_test.clone()
        X_test_perturbed[:, :, feature_index] = 0  # 扰动指定特征
        perturbed_predictions = model(X_test_perturbed).cpu().numpy()
        sensitivity = np.abs(original_predictions - perturbed_predictions).mean()
    return sensitivity

# 计算每个特征的敏感性
feature_importances = []
for i in range(len(features)):  # 现在考虑 AQI 以外的特征
    sensitivity = sensitivity_analysis(model, X_test, i)
    feature_importances.append((features[i], sensitivity))

# 显示特征重要性
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print("特征重要性：")
for feature, importance in feature_importances:
    print(f"{feature}: {importance:.4f}")
