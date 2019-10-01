
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
import os.path
import json
from keras.models import model_from_json
import matplotlib.pyplot as plt

import seaborn as sns

seed = 124
np.random.seed(seed)



df= pd.read_excel(r'H:\ML\hospital\heart1.xlsx')

sns.pairplot(df)

sns.pairplot(df, hue="a1p2")


sns.pairplot(df, vars=["age","cpt","rbp","a1p2"], hue="sex")


sns.pairplot(df, vars=["age", "cpt","rbp"], diag_kind="kde")


sns.pairplot(df, vars=["age","cpt","rbp"],kind="reg")




data = {
    'A': [*np.random.random(5)],
    'B': [*np.random.random(5)],
    'C': ['X', 'Y', 'X', 'X', 'Y']
}

df2 = pd.DataFrame(data)

sns.set(style="ticks", color_codes=True)
sns.pairplot(df, hue='C')
