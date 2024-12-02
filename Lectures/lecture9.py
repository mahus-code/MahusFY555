from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
Y = iris.target

# split into test/train for handin

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state = 42)
tree.fit(x,Y)

# Now visualize:

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, filled = True, rounded=True, class_names =['Setos', 'Versicolor', 'Virginica'], 
                           feature_names=['Sepal length', 'Sepal width', 'petal length', 'petal width'], out_file = None)
print('done')
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')