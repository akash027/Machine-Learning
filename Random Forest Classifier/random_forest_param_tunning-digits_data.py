from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_predict



dataset  = datasets.load_digits()

images_features = dataset.images.reshape((len(dataset.images,), -1))
images_targets = dataset.target


random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

feature_train, feature_test, target_train, target_test = train_test_split(images_features, images_targets, test_size=0.3)

param_grid = {
    "n_estimators" : [10, 100, 500, 1000],
    "max_depth" : [1,5,10,15],
    "min_samples_leaf" : [1,2,3,4,5,10,20,30,40,50]
    }


grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)

grid_search.fit(feature_train, target_train)

print(grid_search.best_params_)

optimal_estimator = grid_search.best_params_.get("n_estimators")
optimal_depth = grid_search.best_params_.get("max_depth")
optimal_leaf = grid_search.best_params_.get("min_samples_leaf")


best_model = RandomForestClassifier(n_estimators=optimal_estimator,
                                    max_depth=optimal_depth,
                                    min_samples_leaf=optimal_leaf)

k_fold = KFold(n_split=10, random_state=123)


predictions = cross_val_predict(best_model, feature_test, target_test, cv=k_fold)

print("Accuracy of the tuned model: ", accuracy_score(target_test,predictions))