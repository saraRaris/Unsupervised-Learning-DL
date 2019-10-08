import numpy as np
import os
import json
import pickle
import sys, argparse
import cv2

from keras import Model
from keras.models import model_from_json
from keras.applications.resnet50 import preprocess_input

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt



def parse_args():
    
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Unsupervised learning using CNNs.')
    parser.add_argument('--model',default='ResNet50', type=str)
    parser.add_argument('--clustering_method',default='Kmeans', type=str)
    parser.add_argument('--loss',default=None, type=str)
    args = parser.parse_args()
    
    return args



def data_preparation(test_path):

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

    #Convert to float
    x_test = x_test.astype('float32')
    
    #Normalize inputs from [0, 255] to [0, 1]
    x_test = x_test / 255

    return x_test, y_test



def load_model(model_name, loss):
    
    if model_name != 'ResNet50' and model_name != 'Xception' and model_name != 'InceptionResNetV2':
        print('Model name unknown, please choose ResNet50, Xception or InceptionResNetV2.')
        sys.exit()
    
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



def visualize_data(features, labels, num_clusters):
    ''' This function applied PCA to the features to reduce dimensionality and plots them '''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    features_tsne = tsne.fit_transform(features)
    fig = plt.figure()
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    plt.show()



def print_vectors_feats(feats, labels):
    ''' This function plots the normalized features for each class '''
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



def get_centers(x_train, y_train, model):
    ''' This function computes the centers of the clusters and saves them in three separate files '''
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
    test_path = 'dataset/test/'
    num_clusters = 3
    args = parse_args()
    
    print('Preparing data...')
    x_test, y_test = data_preparation(test_path)
    

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


