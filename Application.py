import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report

url = "heart.csv"

dataset = pd.read_csv(url)

print(dataset.isnull().sum())

print(dataset.head())

dataset.info()


x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


Scaler = MinMaxScaler(feature_range=(0, 1))
x = Scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=False)

# svm
svm = svm.SVC(kernel='rbf')
svm.fit(x_train, y_train)

# knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


y_svm = svm.predict(x_test)

y_knn = knn.predict(x_test)


print(accuracy_score(y_test, y_svm)*100)
print(classification_report(y_test, y_svm))

plot_confusion_matrix(svm, x_test, y_test)
plt.show()

print(accuracy_score(y_test, y_knn)*100)
print(classification_report(y_test, y_knn))

plot_confusion_matrix(knn, x_test, y_test)
plt.show()
