import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
pd.options.mode.chained_assignment = None
titanic = pd.read_csv('train.csv')
#x = titanic[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
x = titanic[['Pclass','Sex','Age','SibSp','Parch']]
y = titanic['Survived']
x['Age'] = x['Age'].fillna(x['Age'].mean())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)
x_train['Sex'] = x_train['Sex'].map({'male':0,'female':1})
x_test['Sex'] = x_test['Sex'].map({'male':0,'female':1})

classifier = RandomForestClassifier(random_state=10)
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)
print('ACCURACY: ',accuracy_score(y_test, predicted))
scores = cross_val_score(classifier, x_train, y_train, cv=5, scoring='accuracy')
print('ACCURACY(Cross Validation): ',scores)
print('AVERAGE ACCURACY: ',(scores.mean()))

parameters = {'n_estimators':[100],
              'criterion':['gini'],
              'max_depth':[5],
              'min_samples_split':[10],
              'min_samples_leaf':[5],
              'random_state':[0]}

classifier = RandomForestClassifier()
gsearch = GridSearchCV(classifier, parameters, iid = False, cv = 5)
gsearch.fit(x_train, y_train)
model = gsearch.best_estimator_
score = model.score(x_test, y_test)
print('ACCURACY(GridSearchCV): ',score)

