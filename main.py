import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


class RBF_SFM_OCC:


    def __init__(self, gamma=1):
        self.gamma = gamma
        self.wektory_nosne = None
        self.przesuniecie = None

    def fit(self, X):

        # zapisanie liczby probek do zmiennej
        n_probek = X.shape[0]

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

            # wartosc pred jest korygowana o przesuniecie (ale ujemne ???)
            pred -= self.przesuniecie

            # jesli wartosc pred jest wieksza od 0 do listy y_pred dodawana jest wartosc 1
            if pred > 0:
                y_pred.append(True)
            else: y_pred.append(False)

        return np.array(y_pred)


iris = datasets.load_iris(as_frame=True)

# Pobranie wartosci
X = iris.data.values

# Pobranie informacji o gatunkach irysów (jest 3 gatunki)
y = iris.target.values

# Wybranie tylko gatunków 0 i 1, odrzucenie gatunku 2
setosa_i_versicolor = (y == 0) | (y == 1)
X = X[setosa_i_versicolor]
y = y[setosa_i_versicolor]

# walidacja 5 na 2
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)

gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3]
wynikiWlasny = np.zeros(shape=[len(gammas), rskf.get_n_splits()])
wynikiSklearn = np.zeros(shape=[len(gammas), rskf.get_n_splits()])

fig, ax = plt.subplots(len(gammas), 2, figsize=(len(gammas)*2, len(gammas)*3))
fig.suptitle("Wyniki dla własnego SVM")
fig.tight_layout(pad=5)

# Wyniki accuracy dla różnych gamma, dla własnego SVM i OneClassSVM

for i,gamma in enumerate(gammas):

    for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        # trenowanie tylko na klasie ==1
        train_index_1 = train_index[y[train_index] == 1]
        clf = RBF_SFM_OCC(gamma=gamma)
        clf.fit(X[train_index_1])
        y_pred = clf.predict(X[test_index])
        wynikiWlasny[i,j]=accuracy_score(y[test_index], y_pred)

        ax[i, 0].scatter(X[train_index][:, 0], X[test_index][:, 1], c=y[test_index])
        ax[i, 0].scatter(X[train_index][:, 2], X[test_index][:, 3], c=y[test_index])
        ax[i, 0].set_title("Test (gamma = "+ str(gamma)+")")
        ax[i, 1].scatter(X[train_index][:, 0], X[test_index][:, 1], c=y_pred)
        ax[i, 1].scatter(X[train_index][:, 2], X[test_index][:, 3], c=y_pred)
        ax[i, 1].set_title("Predykcja (gamma = "+ str(gamma)+")")

    tekst = "Accuracy_score = " + str(np.round(np.mean(wynikiWlasny[i,:]), 3))
    ax[i, 1].text(0.03, 0.8, tekst, transform=ax[i, 1].transAxes)


    for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        clf = OneClassSVM(kernel='rbf', gamma=gamma)
        clf.fit(X[train_index])
        y_pred = clf.predict(X[test_index])
        wynikiSklearn[i, j] = accuracy_score(y[test_index], y_pred)

plt.savefig("Wykres.png")
plt.close()

# Wyznaczanie średnich dla każdej wartości gamma
meanWlasny = np.mean(wynikiWlasny, axis=1)
stdWlasny = np.std(wynikiWlasny, axis=1, )
meanSklearn = np.mean(wynikiSklearn, axis=1)
stdSklearn = np.std(wynikiSklearn, axis=1)

# Testy na innych klasyfikatorach
clfs = {
"DTC": DecisionTreeClassifier(),
"SVM_Linear": OneClassSVM(kernel="linear"),
"SVM_Poly": OneClassSVM(kernel="poly")
}

scores = np.zeros((len(clfs), 2 * 5))



for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        scores[clf_id, fold_id] = accuracy_score(y[test_index], y_pred)




# Wybór najlepszego wyniku zależnego od gammy dla własnego SVM i OneClassSVM
wiersz_z_naj_srednia = [0 ,0]
for i in range(wynikiWlasny.shape[0]):
    if i == 0:
        najwieksza_srednia = np.mean(wynikiWlasny[i,:], axis=0)
        najwieksza_srednia_OneClassSVM = np.mean(wynikiSklearn[i, :], axis=0)
    else:
        x = np.mean(wynikiWlasny[i,:], axis=0)
        if najwieksza_srednia <= x:
            najwieksza_srednia = x
            wiersz_z_naj_srednia[0] = i
        y = np.mean(wynikiSklearn[i, :], axis=0)
        if najwieksza_srednia_OneClassSVM <= y:
            najwieksza_srednia_OneClassSVM = y
            wiersz_z_naj_srednia[1] = i

scores = np.vstack((scores, wynikiWlasny[wiersz_z_naj_srednia[0]], wynikiSklearn[wiersz_z_naj_srednia[1]]))

# Zmienne do testów t studenta
t_statystki = np.zeros((scores.shape[0],scores.shape[0]))
p = t_statystki.copy()
istot_statyst = np.zeros(t_statystki.shape).astype(bool)
przewaga = istot_statyst.copy()
alpha = 0.05

# Test tstudenta
for i in range(scores.shape[0]):
    for j in range(scores.shape[0]):
        t_statystki[i, j], p[i, j] = stats.ttest_rel(scores[i, :], scores[j, :])
        if np.mean(scores[i,:]) > np.mean(scores[j,:]): przewaga[i, j] = True
        if p[i, j] < alpha: istot_statyst[i, j] = True

przewaga_istot_statyst = przewaga * istot_statyst

with open('ttstudent_wyniki.csv', 'w') as plik:
    plik.write("Macierz t_statystki:\n")
    plik.write(tabulate(t_statystki, headers=["DTC","SVM_Linear","SVM_Poly","SVM_wlasny","OneClassSVM"], floatfmt=".8f"))
    plik.write("\n\nMacierz p:\n")
    plik.write(tabulate(p, headers=["DTC", "SVM_Linear", "SVM_Poly", "SVM_wlasny", "OneClassSVM"]))
    plik.write("\n\nMacierz przewagi (accuracy_score):\n")
    plik.write(tabulate(przewaga, headers=["DTC", "SVM_Linear", "SVM_Poly", "SVM_wlasny", "OneClassSVM"]))
    plik.write("\n\nIstotność statystyczna:\n")
    plik.write(tabulate(istot_statyst, headers=["DTC", "SVM_Linear", "SVM_Poly", "SVM_wlasny", "OneClassSVM"]))
    plik.write("\n\nPrzewaga istotna statystycznie:\n")
    plik.write(tabulate(przewaga_istot_statyst, headers=["DTC", "SVM_Linear", "SVM_Poly", "SVM_wlasny", "OneClassSVM"]))

mean_scores = np.mean(scores, axis=1)
std_scores = np.std(scores, axis=1)

plik = "wyniki.csv"
with open(plik, 'w') as pliczek:
    pliczek.write(tabulate(np.vstack((gammas,meanWlasny,np.round(stdWlasny, 3),meanSklearn,np.round(stdSklearn, 3))).T,headers=["Gamma","Mean (Wlasny SVM)", "Std (Wlasny SVM)","Mean (Sklearn)","Std (Sklearn)"]))
    pliczek.write("\n\nMean(std) DTC(Sklearn)\t\t\t" + str(np.round(mean_scores[0], 3)) + "(" + str(np.round(std_scores[0], 3)) + ")"+ '\n')
    pliczek.write("Mean(std) SVM_Linear(Sklearn)\t" + str(np.round(mean_scores[1], 3)) + "(" + str(np.round(std_scores[1], 3)) + ")"+ '\n')
    pliczek.write("Mean(std) SVM_Poly(Sklearn)\t\t" + str(np.round(mean_scores[2], 3)) + "(" + str(np.round(std_scores[2], 3)) + ")"+ '\n')

# # zbiór testowy syntetyczny
# X, y = make_circles(1000, factor=.1, noise=.1)
#
# # Tworzenie wykresu dla zbioru testowego
# fig, ax = plt.subplots(1, 4, figsize=(20,5))
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
#
# plt.show()
#
# # użycie metody
# svm = RBF_SFM_OCC(gamma=1)
# svm.fit(X_train)
# y_pred = svm.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
#
# # wyswietlenie zbiorów i dokłądności na potrzeby testów
# print("y_train",y_train)
# print("y_test",y_test)
# print("y_pred",y_pred)
# print("accuracy: ",accuracy)

