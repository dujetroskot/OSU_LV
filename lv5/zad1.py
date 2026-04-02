import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure()
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red', label='Klasa 0 (train)')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', label='Klasa 1 (train)')
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='red', marker='x', label='Klasa 0 (test)')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='blue', marker='x', label='Klasa 1 (test)')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci')
plt.show()
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]
 
x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2_boundary = -(theta0 + theta1 * x1_range) / theta2
 
plt.figure()
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red', label='Klasa 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', label='Klasa 1')
plt.plot(x1_range, x2_boundary, 'k-', label='Granica odluke')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke')
plt.show()
 
y_pred = model.predict(X_test)
 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Matrica zabune')
plt.show()
 
print("Tocnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:", recall_score(y_test, y_pred))
 
correct = y_pred == y_test
plt.figure()
plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Tocno')
plt.scatter(X_test[~correct, 0], X_test[~correct, 1], c='black', label='Pogresno')
plt.plot(x1_range, x2_boundary, 'k-', label='Granica odluke')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rezultati klasifikacije na testnom skupu')
plt.show()