# -*- coding: utf-8 -*-
""" ali pourfereydoon 
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
redwankarimsony_heart_disease_data_path = kagglehub.dataset_download('redwankarimsony/heart-disease-data')

print('Data source import complete.')



# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")
df.head()

df.info()

df.describe()

df.describe(include="O")

df.isnull().sum()

df.duplicated().sum()

df['trestbps'] = df['trestbps'].interpolate(method='linear')
df['chol'] = df['chol'].interpolate(method='linear')

df['thalch'] = df['thalch'].fillna(df['thalch'].median())
df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())

df['fbs'] = df['fbs'].fillna(df.apply(lambda row: 1 if row['age'] > 50 else 0, axis=1))
df['exang'] = df['exang'].fillna(df.apply(lambda row: 1 if row['thalch'] < 120 else 0, axis=1))

df['restecg'] = df['restecg'].fillna(df.apply(lambda row: 1 if row['exang']==1 else 0, axis=1))
df['slope'] = df['slope'].fillna(df.apply(lambda row: 2 if row['oldpeak'] > 1 else 1, axis=1))
df['thal'] = df['thal'].fillna(df.apply(lambda row: 7 if row['ca']>0 or row['slope']==2 else 3, axis=1))
df['ca'] = df['ca'].fillna(df.apply(lambda row: 1 if row['thal']==3 or row['exang']==1 else 0, axis=1))

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Age")
plt.show()

plt.figure()
sns.boxplot(x='fbs', y='trestbps', data=df)
plt.title("Trestbps vs Fasting Blood Sugar")
plt.show()

plt.figure()
sns.scatterplot(x='chol', y='thalch', hue='exang', data=df, palette='Set2')
plt.title("Cholesterol vs Thalach colored by Exang")
plt.show()

plt.figure()
sns.countplot(x='thal', palette='magma', data=df)
plt.title("Count of Thalassemia Types")
plt.show()

plt.figure()
sns.barplot(x='fbs', y='chol', data=df, palette='Set3')
plt.title("Average Cholesterol by FBS")
plt.show()

df['fbs'] = df['fbs'].map({True:1, False:0})
df['exang'] = df['exang'].map({True:1, False:0})

sns.countplot(x='thal', hue='sex', data=df, palette='Set2')
plt.title("Thal (Stress Test Result) vs Sex")
plt.xlabel("Thal")
plt.ylabel("Count")
plt.legend(title="Sex")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df[['age','trestbps','chol','thalch','oldpeak','ca','fbs','exang']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap including FBS and Exang")
plt.show()

import pandas as pd

df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")
df = pd.get_dummies(df, drop_first=True)
# Target variable (heart disease)
y = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Feature variables (all except target)
X = df.drop('num', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(float))
X_test = scaler.transform(X_test.astype(float))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))