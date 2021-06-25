# Program to classify images based on Gait entropy image.


# import necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from PIL import Image,ImageOps
from imutils import paths
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import scipy.linalg as la
import cv2


# Path to the directory

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="E:/Biometrics_data/Sonam/LWIR_Ub_All/Gallery_dft_Lowpass/",
	help="path to directory containing the 'Ub_Nm_GalleryGENI' dataset")

ap.add_argument("-d1", "--dataset1", type=str, default="E:/Biometrics_data/Sonam/LWIR_Ub_All/probe_dftct_Lowpass/",
	help="path to directory containing the 'Ub_coat_GENI' dataset1")


ap.add_argument("-m", "--model", type=str, default="knn",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())


models = {
	"knn": KNeighborsClassifier(n_neighbors=1),

}

# Training phase

imagePaths = paths.list_images(args["dataset"])
X_data = []
labels = []


for imagePath in imagePaths:

	image = Image.open(imagePath)
	image=ImageOps.grayscale(image)    
	image=np.array(image)
	X_data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
X_data=np.array(X_data)
n_samples = X_data.shape[0]

pca = PCA()
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1]*X_data.shape[2])
X_centered = X_data - np.mean(X_data, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
evals, evecs = np.linalg.eig(cov_matrix)
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]
variance_retained=np.cumsum(evals)/np.sum(evals)

var_per=.98
index=np.argmax(variance_retained>=var_per)

evecs = evecs[:,:index+1]

reduced_data=np.dot(evecs.T, X_data.T).T

clf=PCA(var_per)
X_train=X_data

X_train=clf.fit_transform(X_train)


le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels)


model = models[args["model"]]

model.fit(X_train,labels)



#Testing Phase

imagePaths1 = paths.list_images(args["dataset1"])
X_data1 = []
label_p = []


for imagePath1 in imagePaths1:
	image1 = Image.open(imagePath1)
	image1=ImageOps.grayscale(image1) 
	image1 = np.array(image1)
	X_data1.append(image1)
	label1 = imagePath1.split(os.path.sep)[-2]
	label_p.append(label1)
X_data1=np.array(X_data1)
n_samples1 = X_data1.shape[0]
pca1 = PCA()

X_data1 = X_data1.reshape(X_data1.shape[0],X_data1.shape[1]*X_data1.shape[2])

X_test1=clf.transform(X_data1)

le = LabelEncoder()
label_p = le.fit_transform(label_p)
print(label_p)


# Accuracy Prediction

predictions = model.predict(X_test1)    
print(predictions)

score = accuracy_score(label_p, predictions)

print(score)
    
    
  
    















