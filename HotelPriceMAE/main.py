from data_preparation import preprocess_data
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import math


# 文件路径和数据预处理
file_path = 'Hotel Prices.xlsx'
# 预处理数据，得到特征X和目标变量y
X, y = preprocess_data(file_path)

# 将数据转换为Tensor，方便PyTorch处理
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 划分数据集，80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 建立一个神经网络模型
class EnhancedRegressor(nn.Module):
    def __init__(self, input_size):
        super(EnhancedRegressor, self).__init__()
        # 定义第一层，全连接层，输入大小为input_size，输出大小为512
        self.fc1 = nn.Linear(input_size, 512)
        # 批量归一化，512个神经元
        self.bn1 = nn.BatchNorm1d(512)
        # Leaky ReLU激活函数
        self.relu1 = nn.LeakyReLU()
        # Dropout层，防止过拟合，丢弃50%的神经元
        self.dropout1 = nn.Dropout(0.5)
        # 第二层，全连接层，输出大小为256
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.5)
        # 第三层，全连接层，输出大小为128
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.5)
        # 第四层，全连接层，输出大小为64
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(0.5)
        # 最后一层，全连接层，输出大小为1
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        # 前向传播，依次通过每一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        return self.fc5(x)


# 实例化模型，输入大小为训练数据的特征数量
model = EnhancedRegressor(X_train.shape[1])
# 初始化模型参数，使用Xavier初始化方法
model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

# 定义损失函数为均方误差
criterion = nn.MSELoss()
# 定义优化器为AdamW，学习率为0.0005
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
# 学习率调度器，监控损失值，损失不再下降时减少学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 训练模型200个epoch
for epoch in range(200):
    model.train()  # 切换到训练模式
    optimizer.zero_grad()  # 清空梯度
    outputs = model(X_train)  # 前向传播
    loss = criterion(outputs, y_train)  # 计算损失
    loss.backward()  # 反向传播
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪，防止梯度爆炸
    optimizer.step()  # 更新参数
    scheduler.step(loss)  # 更新学习率

# 切换损失函数为平均绝对误差，进行最后的微调
criterion = nn.L1Loss()

# 再训练100个epoch
for epoch in range(200, 300):
    model.train()  # 切换到训练模式
    optimizer.zero_grad()  # 清空梯度
    outputs = model(X_train)  # 前向传播
    loss = criterion(outputs, y_train)  # 计算损失
    loss.backward()  # 反向传播
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪
    optimizer.step()  # 更新参数
    scheduler.step(loss)  # 更新学习率

# 评估模型
model.eval()  # 切换到评估模式
with torch.no_grad():  # 不计算梯度
    predictions = model(X_test)  # 预测测试集
    mse = torch.mean((predictions - y_test) ** 2)  # 计算均方误差
    mae = torch.sqrt(mse)  # 计算平方根，得到RMSE

# 计算并打印最后一个epoch的MAE
last_mae = math.sqrt(loss.item())
print(f'MAE: {last_mae:.4f}')
