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
    
    #Convert class vectors to binary class matrices
    n_classes = len(unique_labels(y_train))
    y_train = np_utils.to_categorical(np.array(y_train)-1, num_classes = n_classes)

    return x_train, y_train, x_test, y_test



def train_model(x_train, y_train, batch, x_test, y_test, model_name):
    
    #Define model and other parameters
    if model_name == 'ResNet50':
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
        num_epochs = 15
    elif model_name == 'Xception':
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
        num_epochs = 25
    elif model_name == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (166, 166, 3), pooling = 'avg')
        num_epochs = 25
    else:
        print('Model name unknown, please choose ResNet50, Xception or InceptionResNetV2.')
        sys.exit()

    x = Dense(3, activation='softmax')(model.output)
    model = Model(inputs = model.input, outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    #mc = ModelCheckpoint('weights{epoch:08d}.h5',save_weights_only=True, period=18)
    
    #Sets weights to compensate unbalanced data
    y_ints = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)
    class_weight_dict = dict(enumerate(class_weights.tolist()))
    
    #One hot encoding test set
    y_test = np_utils.to_categorical(np.array(y_test)-1, num_classes = len(unique_labels(y_test)))
    
    #Train model
    history = model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs=num_epochs, batch_size=batch, shuffle = True, class_weight = class_weight_dict)
    
    #Save history
    with open('trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    #Save model
    model_json = model.to_json()
    with open("InceptionResNetV2.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("InceptionResNetV2.h5")
    pdb.set_trace()

    return model



def load_model(model_name):
    
    #Load model structure
    json_file = open('models/'+ model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    #model = VGG16(include_top=False, weights='imagenet', input_shape = (166, 166, 3))
    #x = Flatten(input_shape=model.output.shape)(model.output)
    #x = Dense(4096, activation='relu')(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dense(3, activation='relu')(x)
    #loaded_model = Model(inputs = model.input, outputs=x)

    #for layer in loaded_model.layers:
    #    layer.trainable = False  # freeze layer
    
    #Load weights into new model
    loaded_model.load_weights('models/'+ model_name + '.h5')

    return loaded_model


def feature_extractor(x_test, y_test, model):
    
    #Remove classification layer from model
    layers = [layer.name for layer in model.layers]
    model = Model(inputs = model.input, outputs=model.get_layer(layers[-2]).output)
    
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
    
    #Evaluate cluster accuracy hungarian algorithm
    acc, corresp_labels = cluster_acc(np.array(labels), clustering.labels_)
    print('The accuracy for the clustering is: ' + str(acc*100))

    #final_labels = []
    #for i in clustering.labels_:
    #    final_labels.append(corresp_labels[i])
    #pdb.set_trace()
    #corresp_labels = corresp_labels.tolist()
    #corresp_labels.sort()
    
    #conf_mat = confusion_matrix2(final_labels, labels)
    # Plot normalized confusion matrix
#plot_confusion_matrix(labels, final_labels, classes=np.asarray(corresp_labels), normalize=True, title='Normalized confusion matrix')
#plt.show()

    clustering_labels = clustering.labels_.tolist()

    #Evaluate similarity adjusted rand score
    ars = metrics.adjusted_rand_score(labels, clustering_labels)
    print('Evaluation of similarity with adjusted rand score: ' + str(ars))
    
    #Evaluate similarity normalized_mutual_info_score
    nmi = metrics.normalized_mutual_info_score(labels, clustering_labels, average_method='warn')
    print('Evaluation of similarity with normalized mutual score: ' + str(nmi))



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

    return



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


def activations(test_image, label, model):
    #test_image = cv2.resize(test_image, (224, 224))
    test_image = np.expand_dims(test_image, axis=0)
    pdb.set_trace()
    #layer_outputs = [layer.output for layer in model.layers[:5]]
    num = 0
    layer_outputs = [layer.output for layer in model.layers[8:9] if not layer.name.startswith('input')]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_image)
    pdb.set_trace()
    
    for l in range(0, len(activations)):
        fig=plt.figure()
        columns = 8
        rows = 4
        for i in range(1, columns*rows +1):
            activation = activations[l]
            fig.add_subplot(rows, columns, i)
            plt.imshow(activation[0, :, :, i-1], cmap='viridis')
        fig.suptitle('Plotting layer: '+str(l+num))
        plt.show()
        pdb.set_trace()


def testing_classification(model, x_test, y_test):
    correct = 0
    for i, image in enumerate(x_test):
        #image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        label = model.predict(image)
        if label.tolist()[0].index(max(label[0]))+1 == y_test[i]:
            correct+=1
        
            #print(str(correct) + ' correct out of: ' + str(len(y_test)))
        print('True label: ' + str(y_test[i]), 'Predicted label: '+ str(label.tolist()[0].index(max(label[0]))+1))
    acc = correct/len(x_test)*100
    return acc



def feat_repr(feats, labels):
    class_1 = []
    class_2 = []
    class_3 = []
    pdb.set_trace()
    for i, f in enumerate(feats):
        if labels[i] == 1:
            class_1.append(f[941])
        elif labels[i] == 2:
            class_2.append(f[941])
        else:
            class_3.append(f[941])
    threshold_1 = np.mean(class_1)
    threshold_3 = np.mean(class_3)
    predicted = []
    true = []
    for i, f in enumerate(feats):
        if f[941] > threshold_1 and f[941] < threshold_3:
            predicted.append(i)
            true.append(labels[i])
    pdb.set_trace()


def plot_history(model_name):
    history = pickle.load(open('models/' + model_name +'.p',"rb"))
    
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
    num_clusters = 3
    args = parse_args()
    
    print('Preparing data...')
    x_train, y_train, x_test, y_test = data_preparation(train_path, test_path)
    
    print('Training begins...')
    model = train_model(x_train, y_train, batch, x_test, y_test, args.model)
    #plot_history(args.model)
    
    print('Loading model...')
    model = load_model(args.model)

    #print('Show activations')
    #activations(x_test[0], y_test[0], model)
    
    print('Data features are now begining to extract...')
    data_features, labels = feature_extractor(x_test,y_test, model)
    
    #Visualizing the data
    visualize_data(data_features, labels, num_clusters)
    
    print('Clustering begining...')
    clustering(data_features, labels, num_clusters, args.clustering_method)


