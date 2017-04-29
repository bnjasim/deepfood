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

# path to the model weights files.
# weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_256.h5'

epochs = 50
batch_size = 16

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150))
print('VGG Model loaded')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(8, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)
print 'Top Model weights Loaded'
# add the model on top of the convolutional base

last = model.output

x = Flatten()(last)
x = Dense(1024, activation='relu')(x)
preds = Dense(200, activation='softmax')(x)

model = Model(initial_model.input, preds)


model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

# fine-tune the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, y_test))


