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


def initXGBoost(xg):
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
    month_map = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    day_map = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
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

    xg.fit(X_train, y_train)
    print(" XG boost: {: .2f}%".format(xg.score(X_test, y_test)*100))

def prediction(xg, inputs):
    data= np.array(inputs, ndmin=2)
    prediction = xg.predict(data)
    if prediction == 1:
        print("hello")
        return 'accepted.'
    else: 
        print("coucou")
        return 'refused.'