#!python

import numpy as np
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, rmsprop
from keras.utils import np_utils

# Load training data
X_train, Y_train = [], []
for filename in os.listdir('Resized/train/idli'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/idli/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([1, 0, 0, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/porotta'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/porotta/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 1, 0, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/sadya'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/sadya/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 1, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/Vada'):
    if filename.endswith(".jpeg"): 
        im = Image.open('Resized/train/Vada/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 0, 1, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/pizza'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/pizza/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 0, 0, 1, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/mussels'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/mussels/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 0, 0, 0, 1, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/hamburger'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/hamburger/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 0, 0, 0, 0, 1, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/train/omelette'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/train/omelette/'+filename)
        X_train.append(np.asarray(im))
        Y_train.append([0, 0, 0, 0, 0, 0, 0, 1])
        im.close()
    else:    
        continue
        

x_train = np.array(X_train)/255.0
y_train = np.array(Y_train)


# Load Testing data
X_test, Y_test = [], []
for filename in os.listdir('Resized/test/idli'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/idli/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([1, 0, 0, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/porotta'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/porotta/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 1, 0, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
    
for filename in os.listdir('Resized/test/sadya'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/sadya/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 1, 0, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/Vada/'):
    if filename.endswith(".jpeg"): 
        im = Image.open('Resized/test/Vada/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 0, 1, 0, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/pizza/'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/pizza/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 0, 0, 1, 0, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/mussels/'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/mussels/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 0, 0, 0, 1, 0, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/hamburger/'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/hamburger/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 0, 0, 0, 0, 1, 0])
        im.close()
    else:    
        continue
        
for filename in os.listdir('Resized/test/omelette/'):
    if filename.endswith(".jpg"): 
        im = Image.open('Resized/test/omelette/'+filename)
        X_test.append(np.asarray(im))
        Y_test.append([0, 0, 0, 0, 0, 0, 0, 1])
        im.close()
    else:    
        continue
        
x_test = np.array(X_test)/255.0
y_test = np.array(Y_test)


# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=1, validation_data=(x_test, y_test))


# Evaluate
y_class = model.predict_classes(x_test, verbose=1)
count = 0
for a,b in zip(y_test, y_class):
    if (a.argmax() == b):
        count += 1

print 'Test Accuracy = ' + str(count*1.0/len(y_test))

y_class = model.predict_classes(x_train, verbose=1)
count = 0
for a,b in zip(y_train, y_class):
    if (a.argmax() == b):
        count += 1

print 'Train Accuracy = ' + str(count*1.0/len(y_train))



