import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv('2/stocks_closing_prices.csv')
plt.plot(df)
plt.show()

# df.drop(['Day Sequence','Weekdays'],1)
# ds = df.values
# ds = ds.astype('float32')
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds = scaler.fit_transform(ds)
# print(ds)
#
# train_size = int(len(ds) * 0.67)
# test_size = len(ds) - train_size
# train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
# print(len(train), len(test))
#
# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return np.array(dataX), np.array(dataY)
#
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# print(trainX)
#
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print(trainX)
#
# model = Sequential()
# model.add(LSTM(4, input_dim=look_back))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=2)
#
# # with open("nn.pickle","wb") as f:
# #      pickle.dump(model, f)
# #
# # pickle_in = open("nn.pickle","rb")
# # model = pickle.load(pickle_in)
#
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
#
# # trainPredict = scaler.inverse_transform(trainPredict)
# # trainY = scaler.inverse_transform([trainY])
# # testPredict = scaler.inverse_transform(testPredict)
# # testY = scaler.inverse_transform([testY])
#
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
