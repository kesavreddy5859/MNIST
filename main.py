# Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time, sleep
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
start = time()
# Dataset
print('Training data')
print('Folder name: trainDigits')
print('Path: ./trainingDigits')
print('Training examples: ', len(os.listdir('./trainingDigits')))
print('\nTesting data')
print('Folder name: testDigits')
print('Path: ./testDigits')
print('Testing examples: ', len(os.listdir('./testDigits')))
'''
Convert the binary image to a vector representation
Example:
image.txt
00000
00110
01100
Vector representation:
image vector
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
'''
# img2vector
# Args
# filename: file name
# Return
# returnVect: vector representation of image
def img2vector(filename):
returnVect = []
with open('./trainingDigits/' + filename) as f:
lines = f.readlines()
for line in lines:
line = list(map(int, list(line.strip())))
returnVect.append(line)
returnVect = np.array(returnVect).reshape(1024)
label = int(filename.split('_')[0])
return returnVect, label
# Create training and testing dataset
X_train, Y_train = [], []
X_test, Y_test = [], []
# Training set
print('Generating trainig set')
sleep(2)
for file in tqdm(os.listdir('./trainingDigits')):
data, label = img2vector(file)
X_train.append(data)
Y_train.append(label)
# Testing set
print('Generating testing set')
sleep(2)
for file in tqdm(os.listdir('./testDigits')):
data, label = img2vector(file)
X_test.append(data)
Y_test.append(label)
# KNearestNeighbours
# Generating KNearestNeighbors classifier with K=3
knn_model = KNeighborsClassifier(n_neighbors=3)
# Fitting model with dataset and labels (training data)
knn_model.fit(X_train, Y_train)
# Checking score of trainined model
score = knn_model.score(X_test, Y_test)
print('Accuracy: ', score*100)
predict = knn_model.predict(X_test)
conf = confusion_matrix(Y_test, predict)
plt.figure(figsize=(10, 7))
sns.heatmap(conf, annot=True, fmt='g', cmap = 'Blues')
plt.show()
# Clustering
# Creating clustering model with n_clusters = 10 [0-9]
kmeans_model = KMeans(n_clusters=10, random_state=2019)
# Fitting model with test data
kmeans_model.fit(X_test)
# Countplot of clusered classes
labels = kmeans_model.labels_
plt.figure(figsize=(10, 7))
sns.countplot(labels)
plt.title('Labels countplot')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
# Display images
'''
Convert the binary image to a vector representation
Example:
image.txt
00000
00110
01100
Vector representation:
image vector
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
'''
# show_clustered_images
# Args
# label: label value (default=7)
# Return
# display 49 instances from the cluster
def show_clustered_images(label=7):
plt.figure(figsize=(11, 11))
count = 0
for idx, label in enumerate(labels):
if label == 7:
plt.subplot(7, 7, count+1)
plt.imshow(np.array(X_test[idx]).reshape(32, 32))
plt.xticks([])
plt.yticks([])
count += 1
if count==49:
break
plt.show()
show_clustered_images(7)
# Creating dataset with label 7
X_test_additional = []
for idx, label in enumerate(labels):
if label == 7:
X_test_additional.append(X_test[idx])
# Clustering label = 7 further
# Splitting into 3 classes
kmeans_model_2 = KMeans(n_clusters=3, random_state=2019)
kmeans_model_2.fit(X_test_additional)
# Countplot of additional clusters
labels_new = kmeans_model_2.labels_
plt.figure(figsize=(10, 7))
sns.countplot(labels_new)
plt.title('Additional Labels countplot')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
# Kernel runtime
runtime = time() - start
print('Runtime:', runtime//60, 'mins', round(runtime%60, 2), 's')