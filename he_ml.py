import numpy as np
import pandas as pd
import xgboost
from xgboost import plot_importance
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle

train = pd.read_csv('hackerearth_ml/train_indessa.csv')
test = pd.read_csv('hackerearth_ml/test_indessa.csv')
#print(train.head())

def handle(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert,df[column]))

    return df

train = handle(train)
train.drop(['desc','verification_status_joint'],1,inplace=True)
train.fillna(-999999,inplace=True)
#train.dropna(inplace=True)

test = handle(test)
test.drop(['desc','verification_status_joint'],1,inplace=True)
test.fillna(-999999,inplace=True)
#test.dropna(inplace=True)
print("handling done")

X = np.array(train.drop(['loan_status'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(train['loan_status'])

X_test = np.array(test).astype(float)
X_test = preprocessing.scale(X_test)

# dt = DecisionTreeClassifier()
#clf = AdaBoostClassifier(n_estimators=100, base_estimator=lr,learning_rate=1)
clf = xgboost.XGBClassifier(learning_rate =0.01, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                            colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
clf.fit(X,y)
print("fit done")
with open("xgb.pickle","wb") as f:
     pickle.dump(clf, f)

pickle_in = open("xgb.pickle","rb")
clf = pickle.load(pickle_in)
plot_importance(clf)
plt.show()
print("pickle opened")

predictions = clf.predict_proba(X_test)[:,1]
print(predictions)

submission = pd.DataFrame({
        "loan_status": predictions,
        "member_id": test["member_id"]
})

submission.to_csv("submission.csv", index=False)
