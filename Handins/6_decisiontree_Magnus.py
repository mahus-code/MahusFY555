import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# We use Data from a Drop fitting laboratory exercise
# Data includes three sizes of drops: small, medium, and large
# 2 features are included: surface tension and radius
def loadData():
    demiSmall = np.array([0.060489, 0.069004, 0.06606, 0.071914, 0.049073, 0.07199, 0.05718, 0.052036, 0.059167, 0.070523])
    R0DemiSmall = np.array([0.0007936, 0.00082561, 0.00082556, 0.00073188, 0.00082792, 0.00099374, 0.0008215, 0.0008533, 0.0009028, 0.00084154])
    demiMedium = np.array([0.070725, 0.070502, 0.07199, 0.072052, 0.072459, 0.071823, 0.071038, 0.072105, 0.071078, 0.072882])
    R0DemiMedium = np.array([0.0012633, 0.0012052, 0.00099374, 0.00093379, 0.0012414, 0.0012323, 0.0012672, 0.001242, 0.0012301, 0.0012902])
    demiLarge = np.array([0.071857, 0.066265, 0.072949, 0.073958, 0.073345, 0.07333, 0.072885, 0.07457, 0.07323, 0.073498])
    R0DemiLarge = np.array([0.0014885, 0.0013934, 0.0014472, 0.0014642, 0.0014739, 0.0014777, 0.0014618, 0.0014881, 0.0014898, 0.0014983])
    
    X1 = np.hstack((demiSmall, demiMedium, demiLarge)) # Stack the data column wise
    X2 = np.hstack((R0DemiSmall, R0DemiMedium, R0DemiLarge))
    data = np.vstack((X1, X2)).T # Stack the data row wise and transpose

    y1 = np.zeros(len(demiSmall)) # Create targets for each class
    y2 = np.ones(len(demiMedium))
    y3 = np.ones(len(demiLarge)) * 2
    target = np.hstack((y1, y2, y3))
    return data, target

def main() -> None: 
    X, y = loadData()

    # Split data into training and testing
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Create a random forest
    forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4, random_state=42, max_features=2)
    forest.fit(xtrain, ytrain)
    
    # Plot the first decision tree classifier
    plot_tree(forest.estimators_[0], feature_names=['Surface Tension (Gamma)', 
                                                    'Radius (R0)'], 
                                                    class_names=['Small Drop', 'Medium Drop', 'Large Drop'], 
                                                    filled=True, rounded=True, fontsize=12)
    plt.title('Decision Tree Classification of Small, Medium, Large Drops')
    plt.show()

    # Predict using the xtest and xtrain 
    ypred = forest.predict(xtest)
    ypred_train = forest.predict(xtrain)

    # Print the accuracy score for training and test
    print('Accuracy Score for xTest:', accuracy_score(ytest, ypred))
    print('Accuracy Score for xTrain:', accuracy_score(ytrain, ypred_train))

if __name__ == '__main__':
    main()