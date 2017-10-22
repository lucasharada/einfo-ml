from sklearn.model_selection import KFold, GridSearchCV
from sklearn import neighbors, datasets

# Defining the parameters of the model to be considered
param_grid = {"n_neighbors": [x for x in range(1, 30)]
              ,"weights": ['uniform','distance']
              }

# Loading iris dataset
iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

# Searching for the best params
clf = neighbors.KNeighborsClassifier()
grid_search = GridSearchCV(clf, param_grid=param_grid,cv=KFold(n_splits=10,random_state=0,shuffle=True))
grid_search.fit(iris.data, iris.target)

print(grid_search.best_params_)