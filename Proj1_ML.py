<<<<<<< Updated upstream
=======
<<<<<<< HEAD
>>>>>>> Stashed changes
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, average_precision_score
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd


warnings.filterwarnings('ignore')
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rcParams['figure.figsize'] = (10, 5)

df_1 = pd.read_csv("./data/dataset_1.csv")
df_2 = pd.read_csv("./data/dataset_2.csv")

df_1
df_2

df = df_1

X = df

# X = df.iloc[:, :300]
# y = df.target
X.drop(X.loc[:, X.nunique() == 1], axis = 1, inplace=True)
X.loc[:, X.T.duplicated()].describe()
X.drop(X.loc[:, X.T.duplicated()], axis=1, inplace=True)

# Spójrzmy na macierz korelacji naszych danych
corr = X.corr()
# Widzimy że wiele z nich jest od siebie zależnych, w takim razie musimy
# wyeliminować kolumny o korelacji > 0.7
corr = corr[corr > 0.7]
# Tworzymy liste kolumn zależnych...
dependent_columns = corr.apply(lambda row: row[row > 0].index, axis=1)
to_delete = []
for j in range (len(dependent_columns)):
    for k in dependent_columns[j]:
        if k is not dependent_columns.index[j]:
            if k not in dependent_columns.index[0:j]:
                to_delete.append(k)

# i bierzemy wartosci unikalne, ktore nastepnie usuwamy
to_delete = np.array(to_delete)
to_delete = np.unique(to_delete)   
X.drop(columns = to_delete, axis=1, inplace=True)
# Mozemy ponownie spojrzec na macierz korelacji
# Widzimy, ze wszystkie zmienne mają już zadowalającą korelacje
new_corr = X.corr()

data_mean, data_std = mean(X), std(X)

cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

# =============================================================================
# lower = lower.to_frame()
# upper = upper.to_frame()
# temp = []
# for i in range (len(lower)):
#     for j in range (50000):
#         x = X.iloc[j,i]
#         if x < lower[i] or x > upper[i]:
#             temp.append(j)
# temp = np.unique(temp)
# outliers = [x for x in X if x < lower or x > upper]
# =============================================================================
A = X.copy()
B = X.copy()
C = X.copy()
cols = X.select_dtypes('number').columns 


# =============================================================================
# Opcja 1 - 12485 wywalonych

# plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y);
# X_new = QuantileTransformer().fit_transform(X)
# plt.scatter(X_new[:,0], X_new[:,1], c = y);

# =============================================================================

# =============================================================================
# # Opcja 2 - 12485 wywalonych - metoda jak poprzedni tylko inny kod
# 
# A = A.loc[:, cols]
# A = A[(np.abs(stats.zscore(A)) < 3).all(axis=1)]
# =============================================================================

# =============================================================================
# # Opcja 3 - wszystko wywalone xD - po kwantylach, po chuju bo za dużo zer
# # żeby miało sens to wartosc w quantile musi byc giga duza
# B_sub = B.loc[:, cols]
# lim = np.logical_and(B_sub < B_sub.quantile(0.999999, numeric_only=False),
#                      1 > 0)
# B.loc[:, cols] = B_sub.where(lim, np.nan)
# B.dropna(subset=cols, inplace=True)
# =============================================================================

# =============================================================================
# ok = X.describe()
# 
# pipe = Pipeline([
#     ("scale", QuantileTransformer()),
#     ("model", KNeighborsRegressor(n_neighbors=1))
# mod = GridSearchCV(estimator=pipe,
#                  param_grid={
#                    'model__n_neighbors': [9, 10]
#                  },
#                  cv=3)
# ])
# =============================================================================

# SEKCJA DZIELENIA ZBIORU NA RÓWNE CZĘŚCI <333
from sklearn.neural_network import MLPClassifier
A = X.copy()

Train = A.iloc[:37000,:]
Test = A.iloc[37000:,:]

jedyny = Train[Train.target == 1]
zera = Train[Train.target == 0]
zera = zera.sample(frac = 1, random_state = 8)
zeraOgr = zera.iloc[:5700,:]

zbior = [zeraOgr,jedyny]
zbior = pd.concat(zbior)
zbior = zbior.sample(frac = 1, random_state = 8)

# kwantyle = zbior.quantile(0.95)
# kwantyle = kwantyle[kwantyle == 0]
# zbior = zbior.drop(columns = kwantyle.index)

Train_X_ograniczony = zbior.iloc[:,:99] # here
Train_y_ograniczony = zbior.target
Test_y_ograniczony = Test.target
Test_X_ograniczony = Test.iloc[:,:99]

# X_train_ogr, X_test_ogr, y_train_ogr, y_test_ogr =  train_test_split(
#                     X_ograniczony, y_ograniczony, test_size=0.3, random_state=23)

dtc = DecisionTreeClassifier()
dtc.fit(Train_X_ograniczony,Train_y_ograniczony)

y_pred_ogr2 = dtc.predict(Test_X_ograniczony)
confusion_matrix(Test_y_ograniczony,y_pred_ogr2)
average_precision_score(Test_y_ograniczony,y_pred_ogr2)

clf = svm.SVC(kernel = "rbf")
clf.fit(Train_X_ograniczony,Train_y_ograniczony)
y_pred4 = clf.predict(Test_X_ograniczony)
confusion_matrix(Test_y_ograniczony,y_pred4)
average_precision_score(Test_y_ograniczony,y_pred4)

mlp = MLPClassifier(random_state=1, max_iter=300).fit(Train_X_ograniczony, Train_y_ograniczony)

y_pred3 = dtc.predict(X_test)
confusion_matrix(y_test,y_pred3)
average_precision_score(y_test, y_pred3)

# =============================================================================
# ss_train = QuantileTransformer()
# X_train_ogr = ss_train.fit_transform(X_train_ogr)
# ss_test = QuantileTransformer()
# X_test_ogr = ss_test.fit_transform(X_test_ogr)
# 
# =============================================================================










A = A.iloc[:, :99]
A = A[(np.abs(stats.zscore(A)) < 3).all(axis=1)]
A = pd.merge(A, y, left_index=True, right_index=True)

y = A.target
A = A.iloc[:, :99]

kwantyle = A.quantile(0.95)
kwantyle = kwantyle[kwantyle == 0]
A = A.drop(columns = kwantyle.index)

Dop = A.describe()

mod.fit(A, y);
jd2 = pd.DataFrame(mod.cv_results_)
A.describe()

X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.25, random_state=42)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_train)
confusion_matrix(y_train,y_pred)

y_pred2 = dtc.predict(X_test)
confusion_matrix(y_test,y_pred2)
average_precision_score(y_test, y_pred2)
A.describe()



# =============================================================================
# neight = KNeighborsRegressor(n_neighbors=10)
# neight.fit(X_train,y_train)
# y_pred = neight.predict(X_train)
# confusion_matrix(y_train,y_pred)
# =============================================================================





<<<<<<< Updated upstream
=======
=======
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rcParams['figure.figsize'] = (10, 5)

df_1 = pd.read_csv("./data/dataset_1.csv")
df_2 = pd.read_csv("./data/dataset_2.csv")

df_1
df_2

df = df_1


X = df.iloc[:, :300]
y = df.target
X.drop(X.loc[:, X.nunique() == 1], axis = 1, inplace=True)
X.loc[:, X.T.duplicated()].describe()
X.drop(X.loc[:, X.T.duplicated()], axis=1, inplace=True)

# Spójrzmy na macierz korelacji naszych danych
corr = X.corr()
# Widzimy że wiele z nich jest od siebie zależnych, w takim razie musimy
# wyeliminować kolumny o korelacji > 0.7
corr = corr[corr > 0.7]
# Tworzymy liste kolumn zależnych...
dependent_columns = corr.apply(lambda row: row[row > 0].index, axis=1)
to_delete = []
for j in range (len(dependent_columns)):
    for k in dependent_columns[j]:
        if k is not dependent_columns.index[j]:
            if k not in dependent_columns.index[0:j]:
                to_delete.append(k)

# i bierzemy wartosci unikalne, ktore nastepnie usuwamy
to_delete = np.array(to_delete)
to_delete = np.unique(to_delete)   
X.drop(columns = to_delete, axis=1, inplace=True)
# Mozemy ponownie spojrzec na macierz korelacji
# Widzimy, ze wszystkie zmienne mają już zadowalającą korelacje
new_corr = X.corr()
>>>>>>> 38c007c57303d8a996e4d8ba321668653938dbc1
>>>>>>> Stashed changes
