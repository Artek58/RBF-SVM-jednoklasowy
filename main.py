import arff
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RBF_SFM_OCC:


    def __init__(self, gamma=1):
        self.gamma = gamma
        self.wektory_nosne = None
        self.przesuniecie = None

    def fit(self, X):

        # zapisanie liczby probek do zmiennej
        n_probek = X_train.shape[0]



        # stworzenie  macierzy zer, która posłuży do zapisywania ogległości między próbkami
        macierz_odleglosci = np.zeros((n_probek, n_probek))

        # obliczanie odleglosci euklidesowej pomiedzy kazdym elementem
        for i in range(n_probek):
            for j in range(n_probek):

                #||Xi - Xj||^2
                macierz_odleglosci[i][j] = np.linalg.norm(X[i] - X[j]) ** 2

        # funkcja jądra RBF https://www.pycodemates.com/2022/10/the-rbf-kernel-in-svm-complete-guide.html
        macierz_funkcji_rbf = np.exp(-self.gamma * macierz_odleglosci)


        # wektory nosne, deklaracja
        self.wektory_nosne = []


        for i in range(n_probek):
            self.wektory_nosne.append(X[i])

        # Liczba wektorow nosnych
        n_wektor_nosne = len(self.wektory_nosne)


        # wektor zerowy o wymiarach liczby wektorow nosnych

        # https://stats.stackexchange.com/questions/592273/how-to-understand-the-dual-coef-parameter-in-sklearns-kernel-svm
        self.dual_coef = np.zeros(n_wektor_nosne)


        # w macierzy wektorów nośych dla kazdego wiersza dodajemy wszystkie kolumny ze soba
        for i in range(n_wektor_nosne):
            for j in range(n_wektor_nosne):
                self.dual_coef[i] += macierz_funkcji_rbf[i][j]

        # obliczanie wartosci przesuniecia miedzy macierza funkcji rbf i macierza dual_coef i obliczenie sredniej

        self.przesuniecie = np.mean(np.dot(self.dual_coef, macierz_funkcji_rbf))




    def predict(self, X):

        # lista y_predict
        y_pred = []


        for i in range(X.shape[0]):
            pred = 0

            # dla kazdego rzedu w macierzy wektorow nosnych obliczana jest odleglosc euklidesowa między ||X[i] - wektory_nosne[j]||^2
            # potem sie ja mnozy przez dual_coef
            for j in range(len(self.wektory_nosne)):
                pred += self.dual_coef[j] * np.exp(-self.gamma * np.linalg.norm(X[i] - self.wektory_nosne[j]) ** 2)

            # wartosc pred jest korygowana o przesuniecie
            pred -= self.przesuniecie


            # jesli wartosc pred jest wieksza od 0 do listy y_pred dodawana jest wartosc 1
            if pred > 0:
                y_pred.append(True)
            else: y_pred.append(False)

        return np.array(y_pred)



iris = datasets.load_iris(as_frame=True)

#Pobranie wartosci
X = iris.data.values

#Pobranie informacji o gatunkach irysów (jest 3 gatunki)
y = iris.target.values

#Wybranie tylko gatunków 0 i 1, odrzucenie gatunku 2
setosa_i_versicolor = (y == 0) | (y == 1)
X = X[setosa_i_versicolor]
y = y[setosa_i_versicolor]

# walidacja 5 na 2
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=100)

#Wyniki accuracy
wynik = []

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):

    #trenowanie tylko na klasie ==1
    train_index_1 = train_index[y[train_index] == 1]

    # pomoc <- wyswietlanie zbiorów uczacych i testowych
    # print("X[train_index_1]",X[train_index_1],"y[train_index_1]", y[train_index_1])
    # print("X[test_index]", X[test_index], "y[test_index]", y[test_index])

    clf = RBF_SFM_OCC(gamma=1)
    clf.fit(X[train_index_1])
    y_pred = clf.predict(X[test_index])

    #pobranie wartosci do zapisania do pliku
    if i == 0:
        X_testy = X[test_index]
        y_oryginal = y[test_index]
        y_predykcja = y_pred
    else:
        X_testy = np.concatenate((X_testy, X[test_index]))
        y_oryginal = np.concatenate((y_oryginal, y_pred))
        y_predykcja = np.concatenate((y_predykcja, y_pred))

    #zapisywanie accuracy
    wynik.append(accuracy_score(y[test_index], y_pred))

print("Wyniki: Mean(%.3f)" %np.mean(wynik),"Std(%.3f)"%np.std(wynik))

#zapis do pliku
wyniki = np.column_stack((X_testy, y_oryginal, y_predykcja))
plik = "wyniki.csv"
header="sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), y_test, y_pred"
np.savetxt(plik, wyniki, delimiter=",", header=header, fmt=["%.1f", "%.1f", "%.1f", "%.1f", "%d", "%d"])




# # zbiór testowy syntetyczny
# X, y = make_circles(1000, factor=.1, noise=.1)
#
# # Tworzenie wykresu dla zbioru testowego
# fig, ax = plt.subplots(1, 3, figsize=(15,5))
#
# # pierwsza fig
# ax[0].scatter(X[:,0], X[:,1], c=y)
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# ax[0].set_title("Zbiór - dwie klasy")
# ax[0].set_xlim(-1.2, 1.2)
# ax[0].set_ylim(-1.2, 1.2)
#
# # podział na test i train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
# # wywalenie z X_train i Y_train klasy "0", żeby nauczyć zbiór na klasie = "1"
# X_train = X_train[y_train == 1]
# y_train = y_train[y_train == 1]
#
# # zbior treningowy
# ax[1].scatter(X_train[:,0], X_train[:,1], c=y_train)
# ax[1].set_xlabel("X_train[:,0]")
# ax[1].set_ylabel("X_train[:,1]")
# ax[1].set_title("Zbiór treningowy")
# ax[1].set_xlim(-1.2, 1.2)
# ax[1].set_ylim(-1.2, 1.2)
#
# # zbior testowy
# ax[2].scatter(X_test[:,0], X_test[:,1], c=y_test)
# ax[2].set_xlabel("X_test[:,0]")
# ax[2].set_ylabel("X_test[:,1]")
# ax[2].set_title("Zbiór testowy")
# ax[2].set_xlim(-1.2, 1.2)
# ax[2].set_ylim(-1.2, 1.2)
# #plt.show()
#
# # użycie metody
# svm = RBF_SFM_OCC(gamma=1, nu=0.1)
# svm.fit(X_train)
# y_pred = svm.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
#
# # wyswietlenie zbiorów i dokłądności na potrzeby testów
# print("y_train",y_train)
# print("y_test",y_test)
# print("y_pred",y_pred)
# print("accuracy: ",accuracy)
=======
# zbiór test
X, y = make_circles(1000, factor=.1, noise=.1)

# Tworzenie wykresu dla zbioru testowego
fig, ax = plt.subplots(1, 3, figsize=(15,5))

# pierwsza fig
ax[0].scatter(X[:,0], X[:,1], c=y)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Zbiór - dwie klasy")
ax[0].set_xlim(-1.2, 1.2)
ax[0].set_ylim(-1.2, 1.2)

# podział na test i train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# wywalenie z X_train i Y_train klasy "0", żeby nauczyć zbiór na klasie = "1"
X_train = X_train[y_train == 1]
y_train = y_train[y_train == 1]

# zbior treningowy
ax[1].scatter(X_train[:,0], X_train[:,1], c=y_train)
ax[1].set_xlabel("X_train[:,0]")
ax[1].set_ylabel("X_train[:,1]")
ax[1].set_title("Zbiór treningowy")
ax[1].set_xlim(-1.2, 1.2)
ax[1].set_ylim(-1.2, 1.2)

# zbior testowy
ax[2].scatter(X_test[:,0], X_test[:,1], c=y_test)
ax[2].set_xlabel("X_test[:,0]")
ax[2].set_ylabel("X_test[:,1]")
ax[2].set_title("Zbiór testowy")
ax[2].set_xlim(-1.2, 1.2)
ax[2].set_ylim(-1.2, 1.2)
plt.show()

# użycie metody

svm = RBF_SFM_OCC(gamma=1)

svm.fit(X_train)
y_pred = svm.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)

# wyswietlenie zbiorów i dokłądności na potrzeby testów
print("y_train",y_train)
print("y_test",y_test)
print("y_pred",y_pred)

print("accuracy: ",accuracy)

