import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Example of classification problem via Scikit-learn

#Load the dataset
myfile="iris.csv"

#Load the data array and the category array
train_data=np.loadtxt(myfile, delimiter=',', usecols={0,1,2,3}, skiprows=1)
train_result=np.loadtxt(myfile, dtype='str', delimiter=',', usecols={4}, skiprows=1)

#Split train and test 
X_train, X_test, y_train, y_test = train_test_split(train_data, train_result, test_size=0.25, random_state=10)

#Create a KNN classifier
neigh=KNeighborsClassifier(n_neighbors=2)
#Train
neigh.fit(X_train,y_train)
#Predict 
result=neigh.predict(X_test)
#Compute accuracy
accuracy=metrics.accuracy_score(y_test,result)
print("Accuracy "+str(accuracy))
#Show the confusion matrix
cmatrix=metrics.confusion_matrix(y_test,result)
print(cmatrix)

