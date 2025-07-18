import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# ------------- 1. 数据加载与预处理 -------------
df = pd.read_excel(r"C:\Users\dongx\Desktop\北京-逐日1.xlsx")
df["日期"] = pd.to_datetime(df["日期"])
df.set_index("日期", inplace=True)

# 增加三个环境特征
extra_features = ['平均气温', '平均湿度', '平均风力']
input_features = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h'] + extra_features
target_col = 'AQI'

data_model = df[input_features + [target_col]].copy()
data_model.fillna(method='ffill', inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_model)
scaled_df = pd.DataFrame(scaled_data, index=data_model.index, columns=input_features + [target_col])

# ------------- 2. 构造时序数据（滑动窗口） -------------
def create_sequences(data, time_steps=7, target_index=-1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :len(input_features)])
        y.append(data[i+time_steps, target_index])
    return np.array(X), np.array(y)

target_index = scaled_df.columns.get_loc(target_col)
X, y = create_sequences(scaled_data, time_steps=7, target_index=target_index)

# 划分训练集和验证集（80% 训练，20% 验证）
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ------------- 3. 定义堆叠 LSTM 模型 -------------
class StackedLSTMAQIModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, dropout=0.3):
        super(StackedLSTMAQIModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取序列最后一个时间步的输出
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedLSTMAQIModel(input_size=len(input_features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 学习率调度器，当验证损失不下降时降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)

# ------------- 4. 模型训练（增加总epoch及早停策略） -------------
epochs = 300  # 提高训练轮次
best_val_loss = np.inf
patience = 20   # 若验证损失连续20个epoch不下降则停止训练
trigger_times = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    # 打印训练与验证损失
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 学习率调度器更新（基于验证损失）
    scheduler.step(val_loss)
    
    # 早停策略检测
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        # 可保存最佳模型，此处可添加模型保存代码
        best_model_state = model.state_dict()
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# 加载最佳模型（如果需要）
model.load_state_dict(best_model_state)

# ------------- 5. 模型预测与反归一化 -------------
model.eval()
with torch.no_grad():
    X_val_device = X_val.to(device)
    predictions = model(X_val_device).cpu().numpy()
    y_val_np = y_val.cpu().numpy()

def inverse_transform_AQI(aqi_values):
    dummy = np.zeros((aqi_values.shape[0], len(input_features)))
    combined = np.hstack((dummy, aqi_values))
    inv = scaler.inverse_transform(combined)
    return inv[:, -1]

y_pred_real = inverse_transform_AQI(predictions)
y_val_real = inverse_transform_AQI(y_val_np)


print("\n【模型性能评估】")
print(f"验证集 MSE: {mean_squared_error(y_val_real, y_pred_real):.4f}")
print(f"验证集 RMSE: {np.sqrt(mean_squared_error(y_val_real, y_pred_real)):.4f}")
print(f"验证集 MAE: {mean_absolute_error(y_val_real, y_pred_real):.4f}")

# ------------- 6. 绘制模型预测效果图 -------------
# 取验证集对应的日期（最后 len(y_val_real) 个日期）
# test_dates = scaled_df.index[-len(y_val_real):]

# plt.figure(figsize=(12,6))
# plt.plot(test_dates, y_val_real, label='真实 AQI', color='blue', linewidth=1.5)
# plt.plot(test_dates, y_pred_real, label='预测 AQI', color='red', linestyle='--', linewidth=1.5)
# plt.title('【图1】Stacked LSTM 模型 AQI 预测效果', fontsize=14, fontweight='bold')
# plt.xlabel('日期 (Year-Month)', fontsize=12)
# plt.ylabel('AQI 数值', fontsize=12)
# plt.legend(loc='upper left', fontsize=12)
# plt.grid(True)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# # 旋转 x 轴标签 45 度
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.savefig("LSTM_AQI_prediction.png")
# plt.show()

# def feature_sensitivity_analysis(model, X_sample, feature_names):
#     model.eval()
#     X_sample = X_sample[:100].to(device)  # 取100条样本测试
#     base_output = model(X_sample).cpu().detach().numpy()

#     impacts = []
#     for i in range(X_sample.shape[2]):
#         X_perturbed = X_sample.clone()
#         X_perturbed[:, :, i] *= 1.1  # 对第i个特征加10%
#         perturbed_output = model(X_perturbed).cpu().detach().numpy()
#         delta = np.abs(perturbed_output - base_output).mean()
#         impacts.append(delta)

#     # 可视化
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x=feature_names, y=impacts)
#     plt.title('特征敏感性分析（+10% 扰动）')
#     plt.ylabel('平均预测变化 (ΔAQI)')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # 使用方法：
# feature_sensitivity_analysis(model, X_val, input_features)

# 计算绝对误差与相对误差
absolute_errors = np.abs(y_val_real - y_pred_real)
# 为防止除以0，采用 np.maximum 给 y_val_real 加个极小值
relative_errors = absolute_errors / np.maximum(np.abs(y_val_real), 1e-8) * 100.0  # 单位：%

# 取验证集对应的日期（最后 len(y_val_real) 个日期）
test_dates = scaled_df.index[-len(y_val_real):]

# 绘制绝对误差图和相对误差图
plt.figure(figsize=(14, 10))

# 子图1：绝对误差图
# plt.subplot(2, 1, 1)
# plt.plot(test_dates, absolute_errors, label='绝对误差', color='orange', linewidth=1.5)
# plt.title('测试集上绝对误差图', fontsize=14, fontweight='bold')
# plt.xlabel('日期', fontsize=12)
# plt.ylabel('绝对误差', fontsize=12)
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.legend(loc='upper right', fontsize=12)

# 子图2：相对误差图
# plt.subplot(2, 1, 2)
# plt.plot(test_dates, relative_errors, label='相对误差 (%)', color='green', linewidth=1.5)
# plt.title('测试集上相对误差图', fontsize=14, fontweight='bold')
# plt.xlabel('日期', fontsize=12)
# plt.ylabel('相对误差 (%)', fontsize=12)
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.legend(loc='upper right', fontsize=12)
# print(max(relative_errors))
# print(max(absolute_errors))

# plt.tight_layout()
# plt.savefig("Test_Error_Plot.png")
# plt.show()

# --- 6.1 绘制训练集拟合图像 ---
# 使用模型在训练集上进行预测
model.eval()
with torch.no_grad():
    train_predictions = model(X_train.to(device)).cpu().numpy()
    y_train_np = y_train.cpu().numpy()

# 反归一化处理
y_train_pred_real = inverse_transform_AQI(train_predictions)
y_train_real = inverse_transform_AQI(y_train_np)

# 训练集对应的日期：由于构造序列时每个预测目标位于序列末尾，因此对应的日期为原始数据中从第 time_steps 个日期开始
time_steps = 7  # 与数据构造时保持一致
split = int(len(X) * 0.8)  # 训练集样本数
train_dates = scaled_df.index[time_steps : split + time_steps]

# 绘制训练集上真实 AQI 与预测 AQI 的拟合图像
# plt.figure(figsize=(12, 6))
# plt.plot(train_dates, y_train_real, label='真实 AQI', color='blue', linewidth=1.5)
# plt.plot(train_dates, y_train_pred_real, label='预测 AQI', color='red', linestyle='--', linewidth=1.5)
# plt.title('训练集上 AQI 拟合效果', fontsize=14, fontweight='bold')
# plt.xlabel('日期', fontsize=12)
# plt.ylabel('AQI 数值', fontsize=12)
# plt.legend(loc='upper left', fontsize=12)
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("Train_AQI_Fit.png")
# plt.show()
