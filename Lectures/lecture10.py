from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Random forest:
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=10)
forest.fit(x,y)

# Feature importance:
importance = forest.feature_importances_

# Sort in decending order for plot:
indices = np.argsort(importance)[::-1]
feat_labels = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

final_labels = []
for i in indices:
    final_labels.append(feat_labels[i])

fig, ax = plt.subplots(1,1)

plt.title('Feature importane')
ax.bar(range(x.shape[1]), importance[indices], align = 'center')
ax.set_xticks(range(x.shape[1]))
ax.set_xticklabels(final_labels, rotation = 45)
plt.show()