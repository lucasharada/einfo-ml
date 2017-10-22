from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets

# Defining the number of neighbors to be considered
n_neighbors = 15

# Loading iris dataset
iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

# Holdout: 60 train / 40 test
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Training KNN classifier
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
