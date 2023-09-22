import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=2000,centers=2,n_features=2)
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=5).fit_predict(X)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1])

plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1],c=y_pred)
X, y = make_blobs(n_samples=2000,centers=2,n_features=2,random_state=3)
X, y = make_blobs(n_samples=2000,centers=3,n_features=10,random_state=30)

y_pred = KMeans(n_clusters=5).fit_predict(X)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])	
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1],c=y_pred)

import random
centers=random.randint(1,30)
n_features=random.randint(1,30)
X, y = make_blobs(n_samples=2000,centers=centers,n_features=n_features)

temp=[]
for i in range(1,50):
    model=KMeans(n_clusters=i)
    model.fit(X)
    temp.append(model.inertia_)

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 50) , temp , 'o')
plt.plot(np.arange(1 , 50) , temp , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()