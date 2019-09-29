import glob
import cv2
import numpy as np
import pdb
import os
import json
import pickle
import sys, argparse

from keras import Model
from keras.models import Sequential
from keras.applications import ResNet50, VGG16, InceptionResNetV2
from keras.applications.resnet50 import preprocess_input
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Activation
from keras.layers import Input
from keras.utils import np_utils
from keras.models import model_from_json
from keras. callbacks import ModelCheckpoint

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import class_weight


from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


def parse_args():
    
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Unsupervised learning using CNNs.')
    parser.add_argument('--model',default='ResNet50', type=str)
    parser.add_argument('--clustering_method',default='Kmeans', type=str)
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
            model = load_model(model_name, 'None')
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

    #Option for clustering-loss
    if loss == 'clustering':
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



def load_model(model_name, loss):
    
    #Load model structure
    if loss == 'clustering':
        model_name = model_name + '_clusteringloss'
    json_file = open('models/'+model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    #Load weights into new model
    loaded_model.load_weights('models/'+model_name + '.h5')

    return loaded_model


def feature_extractor(x_test, y_test, model, loss):
    
    #Remove classification layer from model
    if loss == 'clustering':
        model = Model(inputs = model.input, outputs=model.layers[-1].output)
    else:
        model = Model(inputs = model.input, outputs=model.layers[-2].output)
    
    #Obtain features
    data_features = []
    labels = []
    i = 0
    for i, im in enumerate(x_test):
        labels.append(y_test[i])
        #Expand dimensions
        img = preprocess_input(np.expand_dims(im, axis=0))
        #Feed image to the network
        features = model.predict(img)
        #Add to data features
        data_features.append(features.flatten())
        print(i)
    
    return data_features, labels



def clustering(features, labels, clusters, method):
    
    #Cluster with the algorithm defined
    if method == 'Kmeans':
        clustering = KMeans(n_clusters=clusters, random_state=0).fit(features)
    elif method == 'AC':
        clustering = AgglomerativeClustering(n_clusters=clusters).fit(features)
    else:
        print('Please choose a valid clustering technique: Kmeans or AC')
        sys.exit()

    clustering_labels = clustering.labels_.tolist()

    #Evaluate similarity adjusted rand score
    ars = metrics.adjusted_rand_score(labels, clustering_labels)
    print('Evaluation of similarity with adjusted rand score: ' + str(ars))
    
    #Evaluate similarity normalized_mutual_info_score
    nmi = metrics.normalized_mutual_info_score(labels, clustering_labels, average_method='warn')
    print('Evaluation of similarity with normalized mutual score: ' + str(nmi))



def visualize_data(Z, labels, num_clusters):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    plt.show()



def print_vectors_feats(feats, labels):
    zipped = zip(feats, labels)
    n_feats = len(feats[0])
    x = np.linspace(0,n_feats, num=n_feats)
    add_1 = np.zeros(n_feats)
    add_2 = np.zeros(n_feats)
    add_3 = np.zeros(n_feats)
    c1 = 0
    c2 = 0
    c3 = 0
    for f,l in zipped:
        if l == 1:
            add_1 += f
            c1 += 1
        elif l == 2:
            add_2 += f
            c2 += 1
        elif l == 3:
            add_3 += f
            c3 += 1
    add_1 = add_1 / c1
    add_2 = add_2 / c2
    add_3 = add_3 / c3
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, add_1, color='red')
    ax.set_title('Features for class 1')
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(x, add_2, color='green')
    ax1.set_title('Features for class 2')
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(x, add_3, color='blue')
    ax2.set_title('Features for class 3')
    plt.show()



def plot_history(model_name, loss):
    
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



def get_centers(x_train, y_train, model):
    if os.path.exists('center_1.csv'):
        c1_m = np.genfromtxt('center_1.csv', dtype=None)
        c2_m = np.genfromtxt('center_2.csv', dtype=None)
        c3_m = np.genfromtxt('center_3.csv', dtype=None)
    else:
        train = zip(x_train, y_train)
        c1 = []
        c2 = []
        c3 = []
        for idx, x in enumerate(train):
            print('Taking image: '+str(idx))
            img = preprocess_input(np.expand_dims(x[0], axis=0))
            pred = model.predict(img)
            if x[1] == 1:
                c1.append(pred.flatten())
            elif x[1] == 2:
                c2.append(pred.flatten())
            elif x[1] == 3:
                c3.append(pred.flatten())
            else:
                print('Unknown class')
        c1_m = np.mean(np.asarray(c1), axis=0)
        np.savetxt('center_1.csv', c1_m, delimiter=',', fmt='%f')
        c2_m = np.mean(np.asarray(c2), axis=0)
        np.savetxt('center_2.csv', c2_m, delimiter=',', fmt='%f')
        c3_m = np.mean(np.asarray(c3), axis=0)
        np.savetxt('center_3.csv', c3_m, delimiter=',', fmt='%f')

    return c1_m, c2_m, c3_m



if __name__ == "__main__":
    train_path = 'dataset/train/'
    test_path = 'dataset/test/'
    batch = 16
    num_clusters = 3
    args = parse_args()
    
    print('Preparing data...')
    x_train, y_train, x_test, y_test = data_preparation(train_path, test_path)
    
    print('Training begins...')
    train_model(x_train, y_train, batch, x_test, y_test, args.model, args.loss)
    #plot_history(args.model, args.loss)
    
    print('Loading model...')
    model = load_model(args.model, args.loss)
    
    print('Data features are now begining to extract...')
    data_features, labels = feature_extractor(x_test,y_test, model, args.loss)

    #Visualizing the data
    visualize_data(data_features, labels, num_clusters)
    
    print('Clustering the features...')
    clustering(data_features, labels, num_clusters, args.clustering_method)

    print('Show features for the classes...')
    print_vectors_feats(data_features, labels)


