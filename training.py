import numpy as np
import os
import json
import pickle
import sys, argparse
import cv2

from keras import Model
from keras.applications import ResNet50, InceptionResNetV2
from keras.applications.resnet50 import preprocess_input
from keras.applications.xception import Xception
from keras.layers import Dense
from keras.layers import Input
from keras.utils import np_utils
from keras.models import model_from_json

from sklearn.utils.multiclass import unique_labels
from sklearn.utils import class_weight

from matplotlib import pyplot as plt



def parse_args():
    
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Unsupervised learning using CNNs.')
    parser.add_argument('--model',default='ResNet50', type=str)
    parser.add_argument('--loss',default=None, type=str)
    args = parser.parse_args()
    
    return args



def data_preparation(train_path, test_path):
    
    #Reading the training data and training labels
    x_train = []
    y_train = []
    with open(train_path+ 'train_labels.json') as json_file:
        data = json.load(json_file)
        for img in os.listdir(train_path + 'images/'):
            im = cv2.imread(train_path + 'images/'+img)
            #Resize images
            x_train.append(im)
            y_train.append(int(data[img.split('.')[0]]))
    x_train = np.array(x_train)

    #Reading the test data and labels
    x_test = []
    y_test = []
    with open(test_path+ 'val_labels.json') as json_file:
        data = json.load(json_file)
        for img in os.listdir(test_path + 'images/'):
            im = cv2.imread(test_path + 'images/'+img)
            x_test.append(im)
            y_test.append(int(data[img.split('.')[0]]))
    x_test = np.array(x_test)

    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)

    #Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #Normalize inputs from [0, 255] to [0, 1]
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test



def train_model(x_train, y_train, batch, x_test, y_test, model_name, loss):
    
    #Define model and other parameters
    if model_name == 'ResNet50':
        if loss == 'clustering':
            model = load_model(model_name, None)
            model = Model(inputs = model.input, outputs=model.layers[-2].output)
            num_epochs = 35
        else:
            model = ResNet50(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
            num_epochs = 15
    elif model_name == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
        num_epochs = 25
    elif model_name == 'Xception':
        model = Xception(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
        num_epochs = 25
    else:
        print('Model name unknown, please choose ResNet50, Xception or InceptionResNetV2.')
        sys.exit()

    if loss == 'clustering' and model_name != 'ResNet50':
        print('Option not available. Please use ResNet50 when training with clustering loss')
        sys.exit()
    #Option for clustering-loss (Labels are the cluster centers)
    elif loss == 'clustering':
        c1_m, c2_m, c3_m = get_centers(x_train, y_train, model)
        y_train_reg = []
        for y in y_train:
             if y == 1:
                 y_train_reg.append(c1_m)
             elif y == 2:
                 y_train_reg.append(c2_m)
             elif y == 3:
                 y_train_reg.append(c3_m)
        y_test_reg = []
        for y in y_test:
            if y == 1:
                y_test_reg.append(c1_m)
            elif y == 2:
                y_test_reg.append(c2_m)
            elif y == 3:
                 y_test_reg.append(c3_m)
        model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['accuracy'])
        model_name = model_name + '_clusteringloss'
        y_train = np.asarray(y_train_reg)
        y_test = np.asarray(y_test_reg)
        #Train model
        history = model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs=num_epochs, batch_size=batch, shuffle = True)
    #Option for training with non-clustering loss
    else:
        x = Dense(3, activation='softmax')(model.output)
        model = Model(inputs = model.input, outputs=x)
        model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        #Convert class vectors to binary class matrices
        n_classes = len(unique_labels(y_train))
        y_train = np_utils.to_categorical(np.array(y_train)-1, num_classes = n_classes)
        y_test = np_utils.to_categorical(np.array(y_test)-1, num_classes = len(unique_labels(y_test)))
        #Sets weights to compensate unbalanced data
        y_ints = [y.argmax() for y in y_train]
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)
        class_weight_dict = dict(enumerate(class_weights.tolist()))
        #Train model
        history = model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs=num_epochs, batch_size=batch, shuffle = True, class_weight = class_weight_dict)
    
    #Save history
    with open('models/'+model_name+'.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    #Save model
    model_json = model.to_json()
    with open('models/'+model_name+'.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('models/'+model_name+'.h5')



def plot_history(model_name, loss):
    ''' This function plots the training history of the models '''
    if loss == 'clustering':
        model_name = model_name + '_clusteringloss'
    history = pickle.load(open('models/' + model_name +'.p',"rb"))
    pdb.set_trace()
    #Plot accuracy
    x1 = np.linspace(1,len(history['acc']),len(history['acc']))
    y1 = history['acc']
    plt.plot(x1,y1)
    y2 = history['val_acc']
    plt.plot(x1,y2)
    plt.title(model_name+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    
    #Plot loss
    x1 = np.linspace(1,len(history['loss']),len(history['loss']))
    y1 = history['loss']
    plt.plot(x1,y1)
    y2 = history['val_loss']
    plt.plot(x1,y2)
    plt.title(model_name+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()



if __name__ == "__main__":
    train_path = 'dataset/train/'
    test_path = 'dataset/test/'
    batch = 56
    args = parse_args()
    
    print('Preparing data...')
    x_train, y_train, x_test, y_test = data_preparation(train_path, test_path)
    
    print('Training begins...')
    train_model(x_train, y_train, batch, x_test, y_test, args.model, args.loss)
    #plot_history(args.model, args.loss)


