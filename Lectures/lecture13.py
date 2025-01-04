import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

x = load_iris().data
y = load_iris().target

#print(np.unique(y))
y = LabelEncoder().fit_transform(y)
#print(np.unique(y))
y = to_categorical(y)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(units = 8, input_dim = 4, activation='sigmoid'))
model.add(Dense(units = 3, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model:
history = model.fit(x_train, y_train, epochs=20, batch_size = 8, validation_split=0.2)


loss, accuracy = model.evaluate(x_test,y_test)
print('Test accruacy:', accuracy, 'test loss = ', loss)

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