import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
 
labels = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}
 
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor='w',
                    label=labels[cl])
 
df = pd.read_csv("C:/Users/dtros/Desktop/osu/OSU_LV/lv5/penguins.csv")
print(df.isnull().sum())
df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)
df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}, inplace=True)
print(df.info())
 
output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']
 
X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
 
#a
classes, train_counts = np.unique(y_train, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)
x = np.arange(len(classes))
plt.figure()
plt.bar(x - 0.2, train_counts, 0.4, label='Train')
plt.bar(x + 0.2, test_counts, 0.4, label='Test')
plt.xticks(x, [labels[c] for c in classes])
plt.title('Broj primjera po klasi')
plt.legend()
plt.show()
 
#b
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
#c
print("Intercept (theta0 po klasi):", model.intercept_)
print("Koeficijenti (theta po klasi):", model.coef_)
 
#d
plot_decision_regions(X_train, y_train, model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Granica odluke - podaci za ucenje')
plt.legend()
plt.show()
 
#e
y_pred = model.predict(X_test)
 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[labels[i] for i in range(3)])
disp.plot()
plt.title('Matrica zabune')
plt.show()
 
print("Tocnost:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=[labels[i] for i in range(3)]))
 
#f
input_variables_more = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']
X_more = df[input_variables_more].to_numpy()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_more, y, test_size=0.2, random_state=123)
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
print("Tocnost s vise ulaznih velicina:", accuracy_score(y_test2, y_pred2))