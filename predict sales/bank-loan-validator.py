# -*- coding: utf-8 -*-
"""
# Validateur de prêts bancaires

Un total de 20000 demandes de prêts acceptées ou non suivant leurs situations personnelles et professionnelles (age, niveau d'études, métiers, salaire, etc) vont être étudiées afin d'automatiser les futures demandes.
"""

# Commented out IPython magic to ensure Python compatibility.
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# execute  !pip install new_pkg_name for installing new packages

data = pd.read_csv('bank-data-training.csv')

job_map = {'unknown': 0, 'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6, 
           'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11 }
marital_map = {'unknown': 0, 'divorced': 1, 'married': 2, 'single': 3}
education_map = {'unknown': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'illiterate': 5,
             'professional.course': 6, 'university.degree': 7}
default_map = {'unknown': 0, 'no': 1, 'yes': 2}
housing_map = {'unknown': 0, 'no': 1, 'yes': 2}
loan_map = {'unknown': 0, 'no': 1, 'yes': 2}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'apr': 0, 'aug': 1, 'dec': 2, 'jul': 3, 'jun': 4, 'mar': 5, 'may': 6, 'nov': 7, 'oct': 8, 'sep': 9}
day_map = {'fri': 0, 'mon': 1, 'thu': 2, 'tue': 3, 'wed': 4}
poutcome_map = {'failure': 0, 'nonexistent': 1, 'success': 2}
y_map = {'no': 0, 'yes': 1}

data['job'] = data['job'].map(job_map)
data['marital'] = data['marital'].map(marital_map)
data['education'] = data['education'].map(education_map)
data['default'] = data['default'].map(default_map)
data['housing'] = data['housing'].map(housing_map)
data['loan'] = data['loan'].map(loan_map)
data['contact'] = data['contact'].map(contact_map)
data['month'] = data['month'].map(month_map)
data['day_of_week'] = data['day_of_week'].map(day_map)
data['poutcome'] = data['poutcome'].map(poutcome_map)
data['y'] = data['y'].map(y_map)

#Start Building the models
from sklearn.model_selection import train_test_split

#spliting train and test data
x= data.drop(['y'], axis= 1)
y= data.y.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)

#from sklearn.preprocessing import Imputer
#fill_values = Imputer(missing_values=0, strategy = 'mean', axis =0)

from sklearn.impute import SimpleImputer 
fill_values = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

# XG boost

import xgboost
xg = xgboost.XGBClassifier()
xg.fit(X_train, y_train)
print(" XG boost: {: .2f}%".format(xg.score(X_test, y_test)*100))

# Make prediction for a new data entry with XG boost Model (à modifier)

# new_values=[1,	189,	60,	23,	846,	30.1,	0.398,	59,	0.9062]
# data= np.array(new_values, ndmin=2)
# prediction = xg.predict(data)
# if prediction == 1:
#   print("diabetes status: TRUE")
# else: 
#   print("diabetes status: FALSE")

def prediction(inputs):
    data= np.array(inputs, ndmin=2)
    prediction = xg.predict(data)
    if prediction == 1:
        return 'Yes.'
    else: 
        return 'No, sorry. I wish you the best of luck nonetheless.'