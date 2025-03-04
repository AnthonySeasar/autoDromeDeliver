import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer


def preprocess_data(file_path):
    df = pd.read_excel(file_path)

    # 如果数据为空，填充refundable为0
    df['Refundable'] = df['Refundable'].fillna(0)

    # 排除异常的房间面积数据，比如房间大小大于100
    # 如果没有房间大小的数据，直接剔除
    df = df[df['Room Size (m²)'].notna() & (df['Room Size (m²)'] <= 100)]

    # 对数值型数据应用对数变换以处理偏态分布
    df['Number of Ratings'] = np.log1p(df['Number of Ratings'])

    # 数值型数据
    X_numerical = df[
        ['Star Rating (out of 5)', 'Customer Rating (out of 10)', 'Number of Ratings', 'Room Size (m²)', 'Refundable']]
    y = df['Price ($)']

    # 类别型数据
    X_categorical = df[['Hotel', 'State', 'City', 'Room Type']]

    # 独热编码
    encoder = OneHotEncoder()
    X_categorical_encoded = encoder.fit_transform(X_categorical).toarray()

    # 使用 KNN 填充其余的缺失数值数据
    imputer = KNNImputer(n_neighbors=5)
    X_numerical_imputed = imputer.fit_transform(X_numerical)

    # 数据标准化
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical_imputed)

    # 合并数值和类别数据
    X_preprocessed = np.hstack([X_categorical_encoded, X_numerical_scaled])

    return X_preprocessed, y
