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
