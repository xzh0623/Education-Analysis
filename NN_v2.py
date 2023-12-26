import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow import _KerasLazyLoader
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

data = pd.read_csv(r"dataset\numerical_dataset.csv")
data_x = data.drop(columns=["SALARY", "STUDENTID"])
data_y = data.pop("SALARY")

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True)

model.fit(
    X_train_scaled, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping],  # 將EarlyStopping回調加入callbacks列表
    verbose=1)

#訓練好model使用predict預測看看在訓練的model跑的回歸線
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

#抓出全重和偏差
W, b = model.layers[0].get_weights() 
for i in range(len(W)):
    print(f'Parameter {X_train.columns[i]}: \t\t{W[i]}\n\n')

for i in range(len(b)):
    print(f'bias {i+1}: {b[i]}')
# import matplotlib as plt

# plt.plot(train_x,pred) #畫出回歸線
# plt.plot(train_x, train_y, 'o') #畫出原本的點
    
import shap

# 創建一個 SHAP explainer
explainer = shap.Explainer(model.predict, X_train)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)