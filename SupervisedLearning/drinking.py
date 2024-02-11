import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as nmp
df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
from scipy.stats import randint as sp_randint

X = df.drop('DRK_YN', axis=1)
y = df['DRK_YN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Optimization
dt_params = {
    'max_depth': [1, 3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train, y_train)

# Neural Network Optimization
nn_params = {
    'max_iter': [250, 500, 1000,2000, 3000, 4000],
    'hidden_layer_sizes': sp_randint(50, 500),
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01, 0.1]
}
nn_random = RandomizedSearchCV(MLPClassifier(), nn_params, n_iter=10, cv=5, n_jobs=-1)
nn_random.fit(X_train, y_train)

# Boosted Decision Trees Optimization
bdt_params = {
    'n_estimators': [5, 25, 50, 100, 200, 500, 1000],
    'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10]
}
bdt_grid = GridSearchCV(GradientBoostingClassifier(), bdt_params, cv=5, n_jobs=-1)
bdt_grid.fit(X_train, y_train)

# Support Vector Machines Optimization
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}
svm_linear_grid = GridSearchCV(SVC(kernel='linear'), svm_params, cv=5, n_jobs=-1)
svm_rbf_grid = GridSearchCV(SVC(kernel='rbf'), svm_params, cv=5, n_jobs=-1)
svm_linear_grid.fit(X_train, y_train)
svm_rbf_grid.fit(X_train, y_train)

# k-Nearest Neighbors Optimization
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11]
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)

# Get the best parameters and models
dt_best = dt_grid.best_estimator_
nn_best = nn_random.best_estimator_
bdt_best = bdt_grid.best_estimator_
svm_linear_best = svm_linear_grid.best_estimator_
svm_rbf_best = svm_rbf_grid.best_estimator_
knn_best = knn_grid.best_estimator_
print("Best Decision Tree params:", dt_grid.best_params_)
print("Best Neural Network params:", nn_random.best_params_)
print("Best Boosted Decision Trees params:", bdt_grid.best_params_)
print("Best SVM Linear params:", svm_linear_grid.best_params_)
print("Best SVM RBF params:", svm_rbf_grid.best_params_)
print("Best KNN params:", knn_grid.best_params_)
X = df.drop('DRK_YN', axis=1)
y = df['DRK_YN']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
dt = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=20)
nn = MLPClassifier(activation="tanh", alpha=0.01, hidden_layer_sizes=(490,), max_iter=2000)
bdt = GradientBoostingClassifier(n_estimators=50, max_depth=3)
svm_linear = SVC(kernel='linear', C=0.1, gamma=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.001)
knn = KNeighborsClassifier(n_neighbors=11)

models = [dt, nn, bdt, svm_linear, svm_rbf, knn]
model_names = ['Decision Tree', 'Neural Network', 'Boosted Decision Trees', 'SVM Linear', 'SVM RBF', 'KNN']

# Train and evaluate models
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'"Drinking "+{name} - Accuracy: {accuracy:.2f}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{name}_drinking.png')
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=nmp.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Drinking "+title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = nmp.mean(train_scores, axis=1)
    train_scores_std = nmp.std(train_scores, axis=1)
    test_scores_mean = nmp.mean(test_scores, axis=1)
    test_scores_std = nmp.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f'{title}_drinking.png')
    return plt
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=None, scoring="accuracy", n_jobs=None):
    """
    Generate a simple plot of the test and training validation curve.
    """
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = nmp.mean(train_scores, axis=1)
    train_scores_std = nmp.std(train_scores, axis=1)
    test_scores_mean = nmp.mean(test_scores, axis=1)
    test_scores_std = nmp.std(test_scores, axis=1)
    plt.figure()
    plt.title("Drinking "+title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig(f'{title}_drinking.png')

dt_param_name = 'max_depth'
dt_param_range = nmp.arange(1, 11)
plot_validation_curve(dt_best, "Validation Curve (Decision Tree)", X, y, param_name=dt_param_name, param_range=dt_param_range, cv=5)

dt_param_name = 'min_samples_split'
dt_param_range = nmp.arange(2, 21)
plot_validation_curve(dt_best, "Validation Curve (Decision Tree)", X, y, param_name=dt_param_name, param_range=dt_param_range, cv=5)

nn_param_name = 'hidden_layer_sizes'
nn_param_range =  nmp.arange(1, 11)
plot_validation_curve(nn_best, "Validation Curve (Neural Network)", X, y, param_name=nn_param_name, param_range=nn_param_range, cv=5)

bdt_param_name = 'n_estimators'
bdt_param_range = nmp.arange(1, 11)
plot_validation_curve(bdt_best, "Validation Curve (Boosted Decision Trees)", X, y, param_name=bdt_param_name, param_range=bdt_param_range, cv=5)

svm_linear_param_name = 'C'
svm_linear_param_range = [0.1, 1, 10, 100]
plot_validation_curve(svm_linear_best, "Validation Curve (SVM Linear)", X, y, param_name=svm_linear_param_name, param_range=svm_linear_param_range, cv=5)

svm_rbf_param_name = 'C'
svm_rbf_param_range = [0.1, 1, 10, 100]
plot_validation_curve(svm_rbf_best, "Validation Curve (SVM RBF)", X, y, param_name=svm_rbf_param_name, param_range=svm_rbf_param_range, cv=5)

knn_param_name = 'n_neighbors'
knn_param_range = nmp.arange(1, 21)
plot_validation_curve(knn_best, "Validation Curve (KNN)", X, y, param_name=knn_param_name, param_range=knn_param_range, cv=5)

plot_learning_curve(dt_best, "Learning Curve (Decision Tree)", X, y, cv=5, n_jobs=-1)
plot_learning_curve(nn_best, "Learning Curve (Neural Network)", X, y, cv=5, n_jobs=-1)
plot_learning_curve(bdt_best, "Learning Curve (Boosted Decision Trees)", X, y, cv=5, n_jobs=-1)
plot_learning_curve(svm_linear_best, "Learning Curve (SVM Linear)", X, y, cv=5, n_jobs=-1)
plot_learning_curve(svm_rbf_best, "Learning Curve (SVM RBF)", X, y, cv=5, n_jobs=-1)
plot_learning_curve(knn_best, "Learning Curve (KNN)", X, y, cv=5, n_jobs=-1)

plt.show()