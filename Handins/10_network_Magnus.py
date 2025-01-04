import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Code to suppress message being printed otherwise from tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical

# Import our data
data, target = fetch_covtype(return_X_y=True)
#print(np.unique(target))
y = LabelEncoder().fit_transform(target)
y = to_categorical(y)
#print(y)

# 20% test data
xtrain, xtest, ytrain, ytest = train_test_split(data, y, test_size=0.20, stratify=y)

# Standardize our data
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

# Create our neural network
model = Sequential()
model.add(Input(shape=(54,)))
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''Training our network and saving as new_model_layer'''
history = model.fit(xtrain, ytrain, epochs=20, batch_size=16, validation_split=0.2)
model.save('model_layer.keras')

loss, accuracy = model.evaluate(xtest, ytest)

print("Accuracy of trained model on test data:",accuracy, "\nLoss of trained model:", loss )

'''Plotting using code from lecture'''

plt.subplots(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc = 'best')

plt.subplot(1,2,2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.show()
'''Importing our trained model and evaluating with test data'''

from keras.models import load_model
model_ = load_model('model_layer.keras')
loss_, accuracy_ = model_.evaluate(xtest,ytest)

print("Accuracy of imported model on test data:",accuracy_, "\nLoss of imported model:", loss_)