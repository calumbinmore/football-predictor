# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:39:19 2022

@author: Calum Binmore
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.gridspec as gridspec
from tabulate import tabulate


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#get_ipython().run_line_magic('matplotlib', 'inline')

'''import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#X=pd.read_csv('Data\train.csv', index_col='id')
#X=pd.read_csv('A:/AnacondaProjects/footballproject/Data/train.csv', index_col='id')
X=pd.read_csv('Data/train.csv', index_col='id')
XX=X.copy()
XX=XX.head(200)
print(XX)
mlX=XX.copy()
#The outcome of the above is false, which shows there are no null values in the target column
# and therefor all the rows of the data are useful. 
Y=X['target']
X.drop('target', axis=1, inplace=True)
X_train, X_valid, y_train, y_valid=train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
#Have sepeciifed the ratios of the train and validate data inside the train_test_split

pd.set_option('display.max_columns', None)
X.head()