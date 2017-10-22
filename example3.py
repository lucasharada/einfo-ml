from sklearn.model_selection import cross_val_score
from sklearn import neighbors, datasets

# Defining the number of neighbors to be considered
n_neighbors = 15

# Loading iris dataset
iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

# Training KNN classifier
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

# Cross-validation: 10-fold
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))