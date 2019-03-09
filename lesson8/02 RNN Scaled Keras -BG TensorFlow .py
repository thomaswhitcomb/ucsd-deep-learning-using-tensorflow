#################################################
# RNN Using Python and Keras
#
###############################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM

#################################################
# Create dataset for RNN
#
DataList = [[ [i+j] for i in range(5)] for j in range (100)]
type(DataList)
DataList[0:5]
DataList[95:100]

TargetList = [(i+5) for i in range(100)]
type(TargetList)
TargetList[0:5]
TargetList[95:100]

data = np.array(DataList, dtype = float)
data[0:5]
data.shape

target = np.array(TargetList, dtype = float)
target[0:5]
target.shape

###################################################
# Scale the data between 0 and 1
#
dataScale = data/100
dataScale[0:2]

targetScale = target/100
targetScale[0:2]

##################################################
# Split data set
#
#x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(dataScale, targetScale, test_size=0.2, random_state=4)

x_train[0:2]
y_train[0:2]

x_test[0:2]
y_test[0:2]

##################################################
# RNN Model
#
model = Sequential()
# Add the LSTM
model.add(LSTM((1), batch_input_shape=(None,5,1), return_sequences=False))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.summary()

#################################################
# Fit the training data to the model
# Measure the accuracy with test data
#
history = model.fit(x_train, y_train, epochs=500, validation_data=(x_test,y_test))

####################################################
# Predict using Testing data
#
results = model.predict(x_test)

plt.scatter(range(20), results,c='r')
plt.scatter(range(20), y_test, c='g')

####################################################
# Plot the loss Function
#
plt.plot(history.history['loss'])







