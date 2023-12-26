import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow import _KerasLazyLoader

train_data = pd.read_csv("dataset\dataset.csv")
train_x = train_data.drop(columns=["SALARY", "STUDENTID"])
train_y = train_data.pop("SALARY")

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

model = Sequential()
#加入神經層第一層(1維)輸出1維(因為資料和輸出都是一維的)
# model.add(Dense(1,input_shape=(1,))) 

model.add(Dense(units=32, activation='relu', input_dim=train_x.shape[1]))
model.add(Dense(units=1, activation='sigmoid'))  # Adjust for binary or multiclass classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs = 200) 
#訓練好model使用predict預測看看在訓練的model跑的回歸線
pred = model.predict(train_x) 
#抓出全重和偏差
W, b = model.layers[0].get_weights() 
print('Weights=', W, '\nbiases=', b)

# import matplotlib as plt

# plt.plot(train_x,pred) #畫出回歸線
# plt.plot(train_x, train_y, 'o') #畫出原本的點