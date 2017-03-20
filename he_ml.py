import numpy as np
import pandas as pd
import xgboost
from xgboost import plot_importance
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
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
train.drop(['pymnt_plan','desc','title','verification_status_joint'],1,inplace=True)
train.fillna(-99999,inplace=True)
#print(train.head())

test = handle(test)
test.drop(['pymnt_plan','desc','title','verification_status_joint'],1,inplace=True)
test.fillna(-99999,inplace=True)
print("handling done")

X = np.array(train.drop(['loan_status'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(train['loan_status'])

#limit = int(math.ceil(0.01*len(test)))

X_test = np.array(test).astype(float)
X_test = preprocessing.scale(X_test)

# dt = DecisionTreeClassifier()
#clf = AdaBoostClassifier(n_estimators=100, base_estimator=lr,learning_rate=1)
clf = xgboost.XGBClassifier()
clf.fit(X,y)
print("fit done")
with open("xgb.pickle","wb") as f:
     pickle.dump(clf, f)

pickle_in = open("xgb.pickle","rb")
clf = pickle.load(pickle_in)
plot_importance(clf)
plt.show()
print("pickle opened")
# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1,len(predict_me))
#     prediction = clf.predict(predict_me)
#     if i % 500 == 0:
#        print(i)
#     if prediction[0]== y[i]:
#         correct+=1
#
# print("accuracy: ",correct/len(X))



predictions = clf.predict_proba(X_test)
print(predictions[:,1])

submission = pd.DataFrame({
        "member_id": test["member_id"],
        "loan_status": predictions[:,1]
})

submission.to_csv("submission.csv", index=False)
