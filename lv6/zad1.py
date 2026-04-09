import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("C:/Users/dtros/Desktop/osu/OSU_LV/lv6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# =============================================================
# ZADATAK 1
# =============================================================

for K in [5, 1, 100]:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_n, y_train)
    
    y_train_p_knn = knn.predict(X_train_n)
    y_test_p_knn = knn.predict(X_test_n)
    
    print(f"\nKNN (K={K}):")
    print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_knn)))
    print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_knn)))
    
    plot_decision_regions(X_train_n, y_train, classifier=knn)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"KNN K={K} - Tocnost train: {accuracy_score(y_train, y_train_p_knn):.3f}")
    plt.tight_layout()
    plt.show()


# =============================================================
# ZADATAK 2
# =============================================================

k_values = range(1, 51)
cv_scores = []

for K in k_values:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, X_train_n, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_K = k_values[np.argmax(cv_scores)]
print(f"\nOptimalni K (unakrsna validacija): {optimal_K}")
print(f"CV tocnost: {max(cv_scores):.3f}")

plt.figure()
plt.plot(k_values, cv_scores)
plt.xlabel('Broj susjeda K')
plt.ylabel('CV tocnost')
plt.title('Odabir optimalnog K')
plt.tight_layout()
plt.show()


# =============================================================
# ZADATAK 3
# =============================================================

for C, gamma in [(1, 1), (0.1, 0.1), (10, 10), (1, 10)]:
    svm_model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    svm_model.fit(X_train_n, y_train)
    
    y_test_p_svm = svm_model.predict(X_test_n)
    
    print(f"\nSVM RBF (C={C}, gamma={gamma}):")
    print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_svm)))
    
    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"SVM rbf C={C}, gamma={gamma} - Test: {accuracy_score(y_test, y_test_p_svm):.3f}")
    plt.tight_layout()
    plt.show()

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    svm_model = svm.SVC(kernel=kernel, C=1)
    svm_model.fit(X_train_n, y_train)
    y_test_p_svm = svm_model.predict(X_test_n)
    print(f"SVM kernel={kernel}, Tocnost test: {accuracy_score(y_test, y_test_p_svm):.3f}")


# =============================================================
# ZADATAK 4
# =============================================================

param_grid = {
    'C':     [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

svm_base = svm.SVC(kernel='rbf')
grid_search = GridSearchCV(svm_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_n, y_train)

print(f"\nOptimalni parametri SVM: {grid_search.best_params_}")
print(f"Najbolja CV tocnost: {grid_search.best_score_:.3f}")

best_svm = grid_search.best_estimator_
y_test_p_best = best_svm.predict(X_test_n)
print(f"Tocnost najboljeg SVM na testu: {accuracy_score(y_test, y_test_p_best):.3f}")

plot_decision_regions(X_train_n, y_train, classifier=best_svm)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title(f"Najbolji SVM - C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
plt.tight_layout()
plt.show()
