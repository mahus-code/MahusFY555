from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

wine = datasets.load_wine()
x = wine.data
y = wine.target

# Random forest:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

forest = RandomForestClassifier(n_estimators=100, random_state=10)
forest.fit(x,y)

# Feature importance:
importance = forest.feature_importances_

# Sort in decending order for plot:
indices = np.argsort(importance)[::-1]
feat_labels = wine.feature_names
# print(feat_labels)
final_labels = []
feat_labels = ['alcohol', 'malic acid', 'ash', 'alcalinity', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'od280/od315', 'proline']

for i in indices:
    final_labels.append(feat_labels[i])

fig, ax = plt.subplots(1,1)

plt.title('Feature importane of wine data set')
ax.bar(range(x.shape[1]), importance[indices], align = 'center', color='indianred')
ax.set_xticks(range(x.shape[1]))
ax.set_xticklabels(final_labels, fontsize='small', rotation = 35)
ax.set_ylabel('Importance')
plt.show()

''' Comparing important feature selection '''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Extract only the most important feature
x_oneFeat = x[:, wine.feature_names.index('color_intensity')].reshape(-1,1)

xtrain_oneFeat, xtest_oneFeat, ytrain_oneFeat, ytest_oneFeat = train_test_split(x_oneFeat, y, test_size=0.3, random_state=42, stratify=y)
xtrain_all, xtest_all, ytrain_all, ytest_all = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# Preprocessing the data

scOneFeat = StandardScaler()
scOneFeat.fit(xtrain_oneFeat)
xtrain_oneFeat_std = scOneFeat.transform(xtrain_oneFeat)
xtest_oneFeat_std = scOneFeat.transform(xtest_oneFeat)

scAllFeat = StandardScaler()
scAllFeat.fit(xtrain_all)
xtrain_allFeat_std = scAllFeat.transform(xtrain_all)
xtest_allFeat_std = scAllFeat.transform(xtest_all)

ppn_one = Perceptron(max_iter=100, alpha=0.05, random_state=1)
ppn_all = Perceptron(max_iter=100, alpha=0.05, random_state=1)

ppn_one.fit(xtrain_oneFeat_std, ytrain_oneFeat)
ppn_all.fit(xtrain_allFeat_std, ytrain_all)

ypred_one = ppn_one.predict(xtest_oneFeat_std)
ypred_all = ppn_all.predict(xtest_allFeat_std)

print('Accuracy of model trained with most important feature (color intensity):', accuracy_score(ytest_oneFeat, ypred_one))
print('Accuracy of model trained with all features:', accuracy_score(ytest_all, ypred_all))

