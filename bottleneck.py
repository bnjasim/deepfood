import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import os
from PIL import Image

top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 50
batch_size = 16

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

print 'Train Images Loaded'

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

print 'Test Images Loaded'

def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_features_train = model.predict(x_train,  batch_size, verbose=1)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    bottleneck_features_validation = model.predict(x_test, batch_size, verbose=1)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    validation_data = np.load(open('bottleneck_features_validation.npy'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    
    print 'Training Top Model'
    model.fit(train_data, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, y_test))
    
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
print 'Bottleneck features Saved'
train_top_model()
print 'Training Finished'