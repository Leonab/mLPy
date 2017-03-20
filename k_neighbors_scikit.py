import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
#print(accuracy)

example_measure = np.array([[4,2,1,1,2,3,1,2,2],[4,2,2,2,1,3,1,2,2]])
example_measure = example_measure.reshape(len(example_measure),-1)

prediction = clf.predict(example_measure)
print(prediction)
