import glob
import cv2
import numpy as np
import pdb
import os
import json

from keras import Model
from keras.models import Sequential
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Input
from keras.utils import np_utils

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

import resnet_arch


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


def data_preparation(train_path, test_path, dims):
    
    #Reading the training data
    x_train = []
    for img in os.listdir(train_path + 'images/'):
        im = cv2.imread(train_path + 'images/'+img)
        #Resize images
        reshaped = cv2.resize(im, (dims[0], dims[1]))
        x_train.append(reshaped)
    x_train = np.array(x_train)
    #pdb.set_trace()
    #x_train = x_train.reshape((x_train.shape[0], dims[0], dims[1], 1))

    #Reading training labels and resizing images
    y_train = []
        with open(train_path+ 'train_labels.json') as json_file:
            data = json.load(json_file)
    for element in data:
        y_train.append(int(data[element]))
    
    #Reading the test data and resizing images
    x_test = []
    for img in os.listdir(test_path + 'images/'):
        im = cv2.imread(test_path + 'images/'+img)
        reshaped = cv2.resize(im, (dims[0], dims[1]))
        x_test.append(reshaped)
    x_test = np.array(x_test)
    #x_test = x_test.reshape((x_test.shape[0], dims[0], dims[1], 1))

    #Reading test labels
    y_test = []
    with open(test_path+ 'val_labels.json') as json_file:
        data = json.load(json_file)
        for element in data:
            y_test.append(int(data[element]))
    
    #Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #Normalize inputs from [0, 255] to [0, 1]
    x_train = x_train / 255
    x_test = x_test / 255
    
    #Convert class vectors to binary class matrices ("one hot encoding")
    n_classes = len(unique_labels(y_train))
    y_train = np_utils.to_categorical(np.array(y_train)-1, num_classes = n_classes)
    y_test = np_utils.to_categorical(np.array(y_test)-1, num_classes = n_classes)
    
    return x_train, y_train, x_test, y_test



def train_model(x_train, y_train, num_epochs, batch):
    #Define model parameters
    model = ResNet50(include_top=True, weights=None, classes=3)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    pdb.set_trace()
    history = model.fit(x_train,y_train, epochs=num_epochs, batch_size=batch, shuffle = True)
    
    model.save_weights("model.h5")



def feature_extractor(data_path, labels_path, model):
    
    data_features = []
    labels = []
    i = 0
    for im in os.listdir(data_path):
        if i > 100:
            return data_features, labels
        #Get label
        label = im.split('_')[0][-2:]
        if not label:
            labels.append(0)
        else:
            labels.append(int(label))
        
        #Read image
        im = cv2.imread(data_path + '/' + im)
        
        #Resize images
        reshaped = cv2.resize(im, (dims[0], dims[1]))
        
        #Expand dimensions
        img = preprocess_input(np.expand_dims(reshaped.copy(), axis=0))
        
        #Feed image to the network
        features = model.predict(img)
        
        #Add to data features
        data_features.append(features.flatten())
        i += 1
        print(i)
    
    
    return data_features, labels



def clustering(features, labels):
    
    #Clustering with kmeans
    clustering = KMeans(n_clusters=58, random_state=0).fit(features)
    #clustering = AgglomerativeClustering(n_clusters=58).fit(features)
    
    #Evaluate cluster accuracy hungarian algorithm
    acc, corresp_labels = cluster_acc(np.array(labels), clustering.labels_)
    print('The accuracy for the clustering is: ' + str(acc*100))
    pdb.set_trace()
    final_labels = []
    for i in clustering.labels_:
        final_labels.append(corresp_labels[i])
    pdb.set_trace()
    corresp_labels = corresp_labels.tolist()
    corresp_labels.sort()
    
    #conf_mat = confusion_matrix2(final_labels, labels)
    # Plot normalized confusion matrix
    plot_confusion_matrix(labels, final_labels, classes=np.asarray(corresp_labels), normalize=True, title='Normalized confusion matrix')
    plt.show()

    clustering_labels = clustering.labels_.tolist()

    #Evaluate similarity adjusted rand score
    ars = metrics.adjusted_rand_score(labels, clustering_labels)
    print('Evaluation of similarity with adjusted rand score: ' + str(ars))
    
    #Evaluate similarity normalized_mutual_info_score
    nmi = metrics.normalized_mutual_info_score(labels, clustering_labels, average_method='warn')
    print('Evaluation of similarity with normalized mutual score: ' + str(nmi))
    
    pdb.set_trace()
    #Evaluating the results
    print(clustering.labels_)


def cluster_acc(y_true, y_pred):
    '''
        Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
        Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
        this mapping and true labels
        :param y_true: True cluster labels
        :param y_pred: Predicted cluster labels
        :return: accuracy score for the clustering
        '''
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        idx1 = int(y_pred[i])
        idx2 = int(y_true[i])
        w[idx1, idx2] += 1
    ind = linear_sum_assignment(w.max() - w)
    val = []
    for i in range(len(ind[0])):
        val.append(w[ind[0][i],ind[1][i]])
    return sum(val) * 1.0 / y_pred.size, ind[1]



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
        print(cm)
        
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        fmt = '.0f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i,j] > 0.1:
                    ax.text(j, i, format(cm[i, j]*100, fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax



def visualize_data(Z, labels, num_clusters):
    '''
        TSNE visualization of the points in latent space Z
        :param Z: Numpy array containing points in latent space in which clustering was performed
        :param labels: True labels - used for coloring points
        :param num_clusters: Total number of clusters
        :param title: filename where the plot should be saved
        :return: None - (side effect) saves clustering visualization plot in specified location
        '''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    plt.show()



def confusion_matrix2(clusters, classes_gt):
    new_gt = deepcopy(classes_gt)
    l = list(set(classes_gt))
    for i in range(len(classes_gt)):
        for j in range(len(l)):
            if classes_gt[i] == l[j]:
                new_gt[i] = j
    conf_mat = np.zeros([len(set(clusters)), len(set(new_gt))])
    for i in range(len(clusters)):
        conf_mat[clusters[i], new_gt[i]] += 1
    
    return conf_mat

if __name__ == "__main__":
    train_path = 'dataset/train/'
    test_path = 'dataset/test/'
    weights_path = 'imagenet'
    labels_path = 'TsignRecgTrain4170Annotation.txt'
    dims = [224, 224]
    num_epochs = 10
    batch = 40
    #model = load_model(weights_path)
    
    print('Preparing the data...')
    x_train, y_train, x_test, y_test = data_preparation(train_path, test_path, dims)
    
    print('Training begins...')
    model = train_model(x_train, y_train, num_epochs, batch)
    
    print('Data features are now begining to extract...')
    data_features, labels = feature_extractor(data_path,labels_path, model)
    
    #Visualizing the data
    visualize_data(data_features, labels, 58)
    
    print('Clustering begining...')
    clustering(data_features, labels)


