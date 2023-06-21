# Predciting Heart Stroke Using Machine Learning Algorithm

- We are developing this model to diagnose stroke on early stages depending on the features like hypertension, average glucose level, bmi, age so that the preventive measures could be taken or in extreme risk cases start medical treatment.
- Another approach was to utilize this data which is present in every hospital database to anaylze trends and get statistics to be used by government policy-making department so that more attention is given to government hospitals (alloting more health budget, providing advance machinery for the treatment of such cases)

## Libraries ued are as following :

#Importing all essential libraries for project
!pip install pandas_profiling
!jupyter nbextension enable toc2/main

#Libraries used for EDA
import os
import pandas as pd
import numpy as np
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
import pandas_profiling as pp
from tabulate import tabulate

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Libraries for machine learning model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Mounting drive on
drive.mount('/gdrive')
