# Read Me

Under this repository the files for the models trained for performing unsupervised learning on a given dataset along with the script necessary to run the code can be found. Moreover, a pdf in LateX is included describing all the steps followed.

## Instructions
In this repository the following files can be found:

* A .py file : ```clustering.py``` that trains the available models and performs clustering.
* Several .json files under the folder models with the architecture of the models used.
* Several .p  with the training history  of the models trained for the task.

The weights of the models trained for this task can be found [here](https://drive.google.com/open?id=1Baw-NlDgTuTVsh0w41hj3_GYYo_S4fwx). To run the code place them inside the models models.
The dataset can be dowlnoaded [here](https://drive.google.com/file/d/1lm0pGemIVukCAwxcBFl1Bl5keNAi_wbR/view).

To run the first part of the code select one of the three possible models available, the clustering algorithm and the loss and run the following instruction on the terminal:

	python train_model.py --model 'model_name' --clustering 'clustering_algorithm' --loss 'loss_name'

The possible model names are:

* ResNet50
* InceptionResNetV2
* Xception

The possible clustering algorithms  are:

* Kmeans
* AC

The loss is None by default. Only one ResNet50 is available with clustering loss. To run this option enter the following instruction on the terminal:

    python train_model.py --model 'ResNet50' --clustering 'clustering_algorithm' --loss 'clustering'
	
	
