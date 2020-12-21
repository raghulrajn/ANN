
# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_excel("INPUT CSV FILE")
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# # Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN


###################################################################
#Hidden layers are used as reLu inorder to overcome the vanishing gradient problem 
#output layer is sigmoid as result of the test is either pass/fail 
#loss_function as "Binary cross entropy" since the output is binary 
###################################################################

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation
# [1,0,0,0,1,1,0,1900,9,1999,1909.1,89.9,156,92.4,12.9,0.03,32.4,91.3,115.3,46.6,514.2,674,481,115,906,286.8,791.4]


print(ann.predict(sc.transform([[1,	0,0,0,1,1,0,1900,9,1999,1909.1,89.9,156,92.4,12.9,0.03,32.4,91.3,115.3,46.6,514.2,674,481,115,906,286.8,791.4]])))

# Predicting the Test set results
y_pred = ann.predict(X_test)
print("y_pred",y_pred)
# y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
accuracy_score(y_test, y_pred)