# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/titanic"]).decode("utf8"))
print(check_output(["ls", "../input/titanic123"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#traindf=pd.read_csv('../input/titanic/ChanDarren_RaiTaran_Lab2a.csv')
traindf=pd.read_csv('../input/titanic/train.csv')
testdf=pd.read_csv('../input/titanic/test.csv')
#print(traindf.columns.values)
#print(traindf.head(3))
#print(traindf.describe())
#g=sns.FacetGrid(traindf, col='Survived')
#g.map(plt.hist, 'Age', bins=40)
#g.add_legend()
#plt.show()
traindf['Gender'] = traindf['Sex'].map({'female':1, 'male':0}).astype(int)
testdf['Gender'] = testdf['Sex'].map({'female':1, 'male':0}).astype(int)

median_age=np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_age[i,j] = traindf[(traindf['Gender'] == i) & (traindf['Pclass'] == j+1)]['Age'].dropna().median()

for i in range(0,2):
    for j in range(0,3):
        traindf.loc[traindf['Age'].isnull() & (traindf['Gender'] == i) & (traindf['Pclass'] == j+1), 'Age'] = median_age[i,j]
        testdf.loc[testdf['Age'].isnull() & (testdf['Gender'] == i) & (testdf['Pclass'] == j+1), 'Age'] = median_age[i,j]

traindf.loc[ traindf['Age'] <= 6, 'Age'] = 0
traindf.loc[(traindf['Age'] >  6) & (traindf['Age'] <= 16), 'Age'] = 1
traindf.loc[(traindf['Age'] > 16) & (traindf['Age'] <= 36), 'Age'] = 2
traindf.loc[(traindf['Age'] > 36) & (traindf['Age'] <= 64), 'Age'] = 3
traindf.loc[ traindf['Age'] > 64, 'Age'] = 4

testdf.loc[ testdf['Age'] <= 6, 'Age'] = 0
testdf.loc[(testdf['Age'] >  6) & (testdf['Age'] <= 16), 'Age'] = 1
testdf.loc[(testdf['Age'] > 16) & (testdf['Age'] <= 36), 'Age'] = 2
testdf.loc[(testdf['Age'] > 36) & (testdf['Age'] <= 64), 'Age'] = 3
testdf.loc[ testdf['Age'] > 64, 'Age'] = 4

port = traindf['Embarked'].dropna().mode()[0]
traindf['Embarked'] = traindf['Embarked'].fillna(port)
traindf['Embarked'] = traindf['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

testdf['Embarked'] = testdf['Embarked'].fillna(port)
testdf['Embarked'] = testdf['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
testdf['Fare'].fillna(testdf['Fare'].dropna().median(), inplace=True)

traindf = traindf.drop(["PassengerId", "Cabin", "Name", "Sex", "Ticket"], axis=1)

#print(traindf.head(3))
#print(traindf.info())

x_train = traindf.drop("Survived", axis=1)
y_train = traindf["Survived"]
x_test = testdf.drop(["PassengerId", "Cabin", "Name", "Sex", "Ticket"], axis=1)
#print(x_test.info())

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_log = logreg.predict(x_test)
acc_log = logreg.score(x_train, y_train)
print('logreg --> all:', len(y_pred_log), ", survived:", np.sum(y_pred_log), ', core:', acc_log)


svc = SVC()
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
acc_svc = svc.score(x_train, y_train)
print('svc --> all:', len(y_pred_svc), ", survived:", np.sum(y_pred_svc), ', core:', acc_svc)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = knn.score(x_train, y_train)
print('knn --> all:', len(y_pred_knn), ", survived:", np.sum(y_pred_knn), ', core:', acc_knn)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred_gaussian = gaussian.predict(x_test)
acc_gaussian = gaussian.score(x_train, y_train)
print('gaussian --> all:', len(y_pred_gaussian), ", survived:", np.sum(y_pred_gaussian), ', core:', acc_gaussian)

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred_perceptron = perceptron.predict(x_test)
acc_perceptron = perceptron.score(x_train, y_train)
print('perceptron --> all:', len(y_pred_perceptron), ", survived:", np.sum(y_pred_perceptron), ', core:', acc_perceptron)

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred_linear_svc = linear_svc.predict(x_test)
acc_linear_svc = linear_svc.score(x_train, y_train)
print('linear_svc --> all:', len(y_pred_linear_svc), ", survived:", np.sum(y_pred_linear_svc), ', core:', acc_linear_svc)

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred_sgd = sgd.predict(x_test)
acc_sgd = sgd.score(x_train, y_train)
print('sgd --> all:', len(y_pred_sgd), ", survived:", np.sum(y_pred_sgd), ', core:', acc_sgd)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred_decision_tree = decision_tree.predict(x_test)
acc_decision_tree = decision_tree.score(x_train, y_train)
print('decision_tree --> all:', len(y_pred_decision_tree), ", survived:", np.sum(y_pred_decision_tree), ', core:', acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred_random_forest = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = random_forest.score(x_train, y_train)
print('random_forest --> all:', len(y_pred_random_forest), ", survived:", np.sum(y_pred_random_forest), ', core:', acc_random_forest)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
#print(models)
y_pred = y_pred_log + y_pred_svc + y_pred_knn + y_pred_gaussian \
         + y_pred_linear_svc + y_pred_sgd \
         + y_pred_decision_tree
         #+ y_pred_random_forest + y_pred_perceptron 
print(y_pred)

#df = pd.DataFrame(y_pred, columns=['survived'])
#df.loc[df['survived']<5, 'survived'] = 0
#df.loc[df['survived']>=5, 'survived'] = 1
#print(df.head(10))
for i in range(0,len(y_pred)):
    if y_pred[i] < 4:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
print(y_pred)
print(len(y_pred))
print(np.sum(y_pred))

submission = pd.DataFrame({
        "PassengerId": testdf["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('./submission.csv', index=False)
