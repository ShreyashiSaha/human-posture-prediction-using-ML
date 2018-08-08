# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 00:39:21 2018

@author: Shreyashi
"""
import pandas as pd
import numpy as np
from sklearn import model_selection, neighbors,tree
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

d=['a','b','c','d','e','f','g','h']
df= pd.read_csv("C:/Users/Shreyashi/Documents/datasets/ConfLongDemo_JSI.csv",names=d)
df.drop(['d'],axis=1,inplace=True)
#print(df.head())

#df['d']=pd.to_datetime(pd.Series(df['d']))
##df['d'] =  pd.to_datetime(df['d'], format='%d%m%Y:%H:%M:%S:%s')

for col in df.columns.values:
    df[col]=pd.to_numeric(df[col],errors='ignore')
    
df.fillna(0, inplace=True)
#print(df.head())

#converting non-numeric data to numeric data
text_digit_vals = {}
def convert_to_int(val):
    return text_digit_vals[val]

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = sorted(set(column_contents))
            x = 1
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df 

df = handle_non_numerical_data(df)

#print(df.corr()["h"])    
#print(df.head())

#df = df.sample(frac=1).reset_index(drop=True)

X=df.iloc[:,:-1].values
X = MinMaxScaler().fit_transform(X)
y=df.iloc[:,-1].values
#X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.025,random_state=4)


#finding the best value of knn classifier
accuracy_list=[]
for i in range(3, 13):  
    knn = neighbors.KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test,pred_i))

#classification using knn
k=accuracy_list.index(max(accuracy_list))+3    
print("Best K Value:",k)
print("knn accuracy score:",max(accuracy_list))
knn = neighbors.KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification report:")
print(classification_report(y_test,y_pred))


#classification using decision tree
clf_entropy=tree.DecisionTreeClassifier(criterion="entropy", random_state=100,min_samples_leaf=2)
clf_entropy.fit(X_train,y_train)
y_pred=clf_entropy.predict(X_test)
print("Decision Tree Score:",accuracy_score(y_test,y_pred))

conf=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf)
class_rep=classification_report(y_test,y_pred)
print("Classification report:")
print(class_rep)

#clustering using kmeans
#df = df.sample(frac=1).reset_index(drop=True)
X=df.iloc[0:7000,3:6].values
#X = MinMaxScaler().fit_transform(X)
y=df.iloc[:,-1].values
kmeans = KMeans(n_clusters=11)
kmeans.fit(X)

#plotting clusters in 2d
y_kmeans = kmeans.predict(X)
plt.figure(figsize=(15,10))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='r',marker='*', s=250, alpha=0.5)
#plt1=plt.figure(figsize=(10,5))
plt.show()

#plotting clusters in 3d
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1],X[:,2], c=y_kmeans)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

#predicting clustering
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print("Clustering accuracy:",correct/len(X))

