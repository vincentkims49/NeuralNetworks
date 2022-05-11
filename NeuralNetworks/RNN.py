import numpy as num# Contains a variety of mathematical functions, including random number generators, linear algebra procedures, Fourier transforms, and more.
from keras.models import SequentialW# A simple stack of layers with only one input and one output tensor can be modelled using the equential model.
from keras.layers import Dense, SimpleRNN#  does operations on the input and return the output.
from sklearn.preprocessing import MinMaxScaler#Scale each feature to a specific range to transform it.
import matplotlib.pyplot as mpl#A collection of Matplotlib's most useful functions.
from sklearn.metrics import mean_squared_error#Mean squared error regression loss.
import math#Mathematical functions must be applied to any functions that you employ.
from pandas import read_csv#will be used to return a new DataFrame with the data and labels from the csv file.

def create_RNN(hidden_units, dense_units, input_shape, activation):#Create a recurrent neural network to compute a control policy. 
    ourModel = Sequential()#appropriate for a plain stack of layers where each layer has exactly one input
    ourModel.add(SimpleRNN(hidden_units,input_shape=input_shape,activation=activation[0]))#fully-connected RNN where the output from previous timestep is to be fed to next timestep
    ourModel.add(Dense(units=dense_units, activation=activation[1]))#the regular deeply connected neural network layer.
    ourModel.compile(loss='mean_squared_error', optimizer='adam')#Once the model is created, you can config the model with losses and metrics 
    return ourModel # Returns our model
 
demo_ourModel = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])# used as a builder to create RNN model

x1 = demo_ourModel.get_weights()[0]
x2 = demo_ourModel.get_weights()[1]
a1 = demo_ourModel.get_weights()[2]
x3 = demo_ourModel.get_weights()[3]
a2 = demo_ourModel.get_weights()[4]
 # Displaying weights 
print('x1 = ', x1, ' x2 = ', x2, ' a1 = ', a1, ' x3 =', x3, 'a2 = ', a2) # Returns the weights on screen

x = num.array([1, 2, 3])#  returns an array, or any sequence. 
inputX = num.reshape(x,(1, 3, 1))#Gives a new shape to an array without changing its data.
prediction_ourModel = demo_ourModel.predict(inputX)# Model groups layers into an object with training and inference features

z = 2
d0 = num.zeros(z)
d1 = num.dot(x[0], x1) + d0 + a1
d2 = num.dot(x[1], x1) + num.dot(d1,x2) + a1
d3 = num.dot(x[2], x1) + num.dot(d2,x2) + a1
c3 = num.dot(d3, x3) + a2
# Displaying vectors
print('d1 = ', d1,'d2 = ', d2,'d3 = ', d3)# Prints the values of the given vectors

#Displaying predictions
print("Network Prediction", prediction_ourModel)# displays the Network Prediction
print("Computational Prediction", c3)# displays the Computational Prediction


def get_train_test(url, split_percent=0.8):# Quick utility that wraps input validation 
    diff = read_csv(url, usecols=[1], engine='python')#supports optionally iterating or breaking of the file into chunks.
    ourdata = num.array(diff.values.astype('float32'))#  returns an array, or any sequence. 
    ourscaler = MinMaxScaler(feature_range=(0, 1))# Feature transformations are accomplished by scaling each individual feature to a predetermined range. Each feature is scaled and translated separately by this estimator to fit within the specified range.
    ourdata = ourscaler.fit_transform(ourdata).flatten()#transformations are done on individual  data 
    pn = len(data)# returns the number of items in a data
    
    datasplit = int(pn*split_percent)# splits data into the predetermined number
    ourtrain_data = data[range(datasplit)]# training data
    ourtest_data = data[datasplit:]# testing the data
    return ourtrain_data, ourtest_data, ourdata# Returns our test data,trained data and our data

# targets and inputs as Y and X are created here
def get_XY(dat, time_steps):# Return only metrics/values that we will base our predictions 
    inputy = num.arange(time_steps, len(dat), time_steps)#Return evenly spaced values within a given interval
    Y = dat[inputy]#Create and modify a dat repository. 
    inputx = len(Y)# returns the number of items in a data
    X = dat[range(time_steps*inputx)]#Create and modify a dat repository and return evenly spaced values within a given interval
    X = num.reshape(X, (inputx, time_steps, 1)) #Gives a new shape to an array without changing its data.   
    return X, Y # Returns the target Y and inputs X

def create_RNN(hidden_units, dense_units, input_shape, activation):#Create a recurrent neural network to compute a control policy. 
    ourModel = Sequential()#appropriate for a plain stack of layers where each layer has exactly one input
    ourModel.add(SimpleRNN(hidden_units,input_shape=input_shape,activation=activation[0]))#  fully-connected RNN where the output from previous timestep is to be fed to next timestep
    ourModel.add(Dense(units=dense_units, activation=activation[1]))#the regular deeply connected neural network layer.
    ourModel.compile(loss='mean_squared_error', optimizer='adam')#Once the model is created, you can config the model with losses and metrics 
    return ourModel # Returns our model

def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_error = math.sqrt(mean_squared_error(trainY, train_predict))# computes the mean squred root of th mean squred error and trains it
    test_error = math.sqrt(mean_squared_error(testY, test_predict))# computes the mean squred root of th mean squred error and tests it
    # Displaying the Root mean squred error
    print('Train RMSE: %.3f RMSE' % (train_error))# Prints the trained rmse
    print('Test RMSE: %.3f RMSE' %(test_error)) # Prints the tested rmse   

# Displaying a plot of the result
def plot_result(trainY, testY, train_predict, test_predict):# Plots the result
    actualData = num.append(trainY, testY)# adds the items to the end of the list
    predictions = num.append(train_predict, test_predict)# adds trained and test prediction items to the end of the list
    rows = len(actualData)# returns the number of items in a data
    mpl.figure(figsize=(15, 6), dpi=80)# Figure instance supports callbacks through a callbacks attribute 
    mpl.plot(range(rows), actualData)# makes a plot on the actual data
    mpl.plot(range(rows), predictions)# makes a plot on the predicted data
    mpl.axvline(x=len(trainY), color='r')#Add a vertical line across the Axes.
    mpl.legend(['Actual data', 'Predicted data'])#Place a legend on the Axes.
    mpl.xlabel('Observation number ')# adds Parameters on the x axis
    mpl.ylabel('Dataset scaled')# adds Parameters on the y axis
    mpl.title('Actual and Predicted Values.')# adds a title
#dataset usrl 
dataset_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'# this includes a link to the dataset that we will be using
time_steps = 12# rounds in sets of time
train_data, test_data, data = get_train_test(dataset_url)#includes training, fetching and testing our dataset from the url
trainX, trainY = get_XY(train_data, time_steps)#training the inputs and the targets
testX, testY = get_XY(test_data, time_steps)#testing the inputs and the targets

#initializing our Model 
ourModel = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1), activation=['tanh', 'tanh'])# creatin RNN and its dense layers 
ourModel.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)#Running 20 epochs and traing the targets and the inputs

# make predictions
tp = ourModel.predict(trainX)# makes a prediction on trainX
ts = ourModel.predict(testX)# makes a prediction on testX
# Display the error
print_error(trainY, testY, tp, ts)# Show error 
#Displays a graph
plot_result(trainY, testY, tp, ts)# Shows a graph result

