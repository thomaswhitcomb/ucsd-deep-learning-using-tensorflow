#################################################
# RNN Using Python and Keras
#
###############################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

from keras.models import Sequential
from keras.layers import LSTM

scale = 200
seq_size = 8
epochs = 700
#################################################
# Create dataset for RNN
#
DataList = [[ [i+j] for i in range(seq_size)] for j in range (scale)]

TargetList = [(i+seq_size) for i in range(scale)]

data = np.array(DataList, dtype = float)

target = np.array(TargetList, dtype = float)
###################################################
# Scale the data between 0 and 1
#
dataScale = data/scale
print(dataScale[0:2])

targetScale = target/scale
print(targetScale[0:2])

##################################################
# Split data set
#
#x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(dataScale, targetScale, test_size=0.2, random_state=4)

##################################################
# RNN Model
#
model = Sequential()
# Add the LSTM
model.add(LSTM((1), batch_input_shape=(None,seq_size,1), return_sequences=False))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.summary()

#################################################
# Fit the training data to the model
# Measure the accuracy with test data
#
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))

####################################################
# Predict using Testing data
#
results = model.predict(x_test)
plt.scatter(range(len(x_test)), results,c='r')
plt.scatter(range(len(x_test)), y_test, c='g')
plt.savefig("lesson8.1.scatter.png")

####################################################
# Plot the loss Function
#
plt.plot(history.history['loss'])
plt.savefig("lesson8.1.loss.png")

