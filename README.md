# Artificial-Intelligence-this contain all the labs i've performed in my course.

LAB 1
1.	Write a small program in Python to print your CV.
print ("CV");
print ("Name: KHAAN \t")
print ("GPA: 3.8 \t")
print ("Degree: B(Cs) \t")
print ("Institude: Iqra University")

2.	Write a program that takes the month (1…12) as input. Print whether the season is summer, winter, spring or autumn depending upon the input month.
x = int(input ("Enter a month"))
z= x
if x<=3:
    print ("This is Winter Season!")
elif x<=6:
    print ("This is Summer Season")
elif x<=9:
    print ("This is Summer seasoN")
else:
    print ("This is Autum Season")

3.	To determine whether a year is a leap year
x = int(input("Enter a year: "))
if x % 4 == 0 and x % 100 != 0 or x % 400 == 0:
   print("\n Is a leap-year")
else:
    print("\n Is not a leap-year")
    
4.	Write a program that takes a line as input and finds the number of letters and digits in the input
x = input ("Type a sentence:\n")
y=z=0
for s in x:
    if s.isdigit():
        y=y+1
    elif s.isalpha():
        z=z+1
print("No. Of Digits are", y)
print ("No. Of Letters are", z)

lab 2: To study and implement basic algorithms in Python

In this lab, we will familiarize ourselves with functions, classes and other advanced constructs of python.

Lab Tasks:
1. Write a program to generate a dictionary that contains (i,sqrt(i)), where i is an integer between 1 and n. n is a number input by the user.
import math
n = int(input("Enter a value"))
z = {}
for i in range(n+1):
    z[i] = math.sqrt(i)
print (z)

2. Write a simple calculator program using functions add, sub, mul and div. The program should accepts two numbers and an operator and calls the corresponding function to perform the operation.
import math
def add(x,y): 
 return x+y
def mul(x,y):
    return x*y
def sub(x,y):
    return x-y
def div(x,y):
    return x/y
print ("Calculator")
print ("1. Addition")
print ("2. Multiplication")
print ("3. Subtraction")
print ("4. Division")
n = input("Enter choice:")
x = int(input("Enter first number:"))
y = int(input("Enter second number:"))
if n == '1':
    print (x , "+" , y, "=", add(x,y))
elif n == '2':
    print (x, "*" , y, '=', mul(x,y))
elif n == '3':
    print (x, '-',y,'=', sub(x,y))
elif n == '4':
    print (x, '/', y, '=', div(x,y))
else:
    print ("Invalid choice! Try Again.")

3. Write a function that generates a list with values that are square of number between 1 and 20.
def KHAAN ():
    l = []
    for z in range(1,21):
        l.append (z*z)
              print(l)
KHAAN()

4. Define a class named Shape with static method printType. Define methods draw() and area(). Now define two class Rectangle and Triangle. Rectangle has two attributes length and width. The Triangle class has attributes a,b and c.  Override the two methods of shape class. Demonstrate the functionality of class by creating its objects.
class shape:
    def draw(self):
        print ("draw")
    def area(self):
        print ("area")
class Rectangle(shape):
    def __init__(self):
        self.length= 0
        self.width= 0
    def draw(self):
        print ("draw")
    def area(self):
        print ("area")
class Triangle (shape):
    def __init__(self):
        self.a=0
        self.b=0
        self.c=0
    def draw(self):
        print ("draw")
    def area(self):
        print ("area")
@staticmethod
def printType():
    print("Print Type")
s= shape()
r = Rectangle()
t= Triangle()
s.area()
s.draw()
r.area()
r.draw(
t.area()
t.draw()

5. Using recursion, write a program to calculate the reverse of a string.
#Explanation: Recursive function (reverse) takes string pointer (str) as input and calls itself with next location to passed pointer (str+1). Recursion continues this way, when pointer reaches ‘\0’, all functions accumulated in stack print char at passed location (str) and return one by one.
#Time Complexity: O(n)

def rev (str):
    if (str == ""):
        return ""
    else:
        return rev (str[1:]) + str[0]
print (rev("Hello"))

 LAB 3: To study and implement Graph search algorithms in Python

1. In this lab, we are going to implement searching algorithms in Python. There are two popular searching algorithms i.e. Depth First Search (Fig. 3a) and Breadth First Search (Fig 3b). 
DFS(G,v)   ( v is the vertex where the search starts )
         Stack S := {};   ( start with an empty stack )
         for each vertex u, set visited[u] := false;
         push S, v;
         while (S is not empty) do
            u := pop S;
            if (not visited[u]) then
               visited[u] := true;
               for each unvisited neighbour w of u
                  push S, w;
            end if
         end while
      END DFS()
3a: Pseudo-code for Depth First Search

Breadth-First-Search(Graph, root):   
    create empty set S
    create empty queue Q      
    root.parent = NIL
    add root to S
    Q.enqueue(root)                      
    while Q is not empty:
        current = Q.dequeue()
        if current is the goal:
            return current
        for each node n that is adjacent to current:
            if n is not in S:
                add n to S
                n.parent = current
                Q.enqueue(n)
3b: Pseudo-code for Breadth First Search

Fig 3: Pseudo-code for Graph Searching algorithms
Lab Task:
1. Provide the implementation of DFS and BFS algorithms in Python.
import queue

#define the graph

r = [[0,1,1,1,1,0,0,0,0],[0,0,0,0,1,1,0,0,0],[0,0,0,0,0,1,1,0,0],
     [0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

#define the frontier queue
frontier = queue.Queue()

#put the starting node as 1
frontier.put(0)
explored=[]
target = 8

while(True):
    #pick the first element from the queue

    if(frontier.empty()):
        print("The target not found")
        break
    n= frontier.get()
    if (n in explored):
        continue
    explored.append(n)
    if (n==target):
        break
    #now get the child of n
    for i in range(0,9):

        if(r[n][i] == 1 and i not in explored):
             frontier.put(i)
print(explored)


Lab 4: To study and understand numpy library

In this lab, we are going to explore numpy. NumPy is an acronym for "Numeric Python" or "Numerical Python". It is an open source extension module for Python, which provides fast precompiled functions for mathematical and numerical routines.


1. Open the Python Notebook provided with this lab and perform the tasks.

•	ARRAY
import numpy as np
b = np.array ([(1,2,3,4), (5,6,7)])
print (b)

•	NUMPY
import numpy as np
a=np.arange(15)
a = a.reshape(3,5)
print(a.shape)

•	RESHAPE
import numpy as np
a = np.arange(18)
a = a.reshape(3,3,2)
print("Number Dimention", a.ndim)
print("Datatype", a.dtype)
print("Shape" , a.shape)
print("Item Size", a.itemsize)
print("Size", a.size)

•	ZEROS
import numpy as np 
a=np.zeros((4,5)) 
print("Zeros", a)
b=np.ones((3,4))
print("Ones",b ) 
b=np.ones((3,3))
b= b*6
print("Six", b)
b= np.linspace(0,2,9)
print("Linspace", b) 
c=np.sin(b)
print("Sin", c) 
c=np.cos(b)
print("Cos", b)

Lab 6:  ANN

1. KERAS
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(0)
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x=dataset[:,0:8]
y=dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(x)
rounded = [round(z[0]) for z in predictions]
print(rounded)

2. DIVIDING DATA INTO TESTING AND TRAINING 
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(0)
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x=dataset[0:538,0:8]
y=dataset[0:538,8]
tx=dataset[538:,0:8]
ty=dataset[538:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(50,activation='relu'))
model.add(Dense(60,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
scores = model.evaluate(tx, ty)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(x)
rounded = [round(z[0]) for z in predictions]
print(rounded)

3. CS
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(0)
dataset = np.genfromtxt("cs-training.csv", delimiter=",")
x=dataset[:10000,2:10]
y=dataset[:100000,1]
tx=dataset[10000:150000,2:10]
ty=dataset[10000:150000,1]
model = Sequential()
model.add(Dense(30,input_dim=8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
scores = model.evaluate(tx, ty)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(x)
rounded = [round(z[0]) for z in predictions]
print(rounded)

LAB 7: CNN

1. GRAPH
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
pyplot.show()

2. CNN
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(5, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

epochs = 2
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=200)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

LAB 8: RNN

1. GRAPH
import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()

2. LSTM
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

LAB 9: DJANGO

1. Use django

#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    execute_from_command_line(sys.argv)
