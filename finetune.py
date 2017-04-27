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