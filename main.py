import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle

# df=pd.read_csv('multiple_linear_regression_dataset.csv')
#
# X = df[['age','experience']]
# y = df['income']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=1)
#
# ppn = Perceptron(max_iter=200,eta0=0.1, random_state=1)
# #default max_iter is 100, now we try with 200, the result doesn't imporve
# ppn.fit(X_train, y_train)
#
# y_pred_ppn = ppn.predict(X_test)
# print('Misclassified examples: %d' % (y_test != y_pred_ppn).sum())
#
# cm = confusion_matrix(y_test, y_pred_ppn)
#
# #create model
# #LR_model=LogisticRegression()
# LR_model = LogisticRegression(max_iter=200, random_state=1, solver='liblinear')
# LR_model.fit(X_train, y_train)
#
#
# import joblib
# # Save RL_Model to file in the current working directory
#
# joblib_file = "income_lr_model.pkl"
# joblib.dump(LR_model, joblib_file)

# Title of your web app
#
# Load your machine learning model
import joblib


with open('income_lr_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

LR_Model = joblib.load("income_lr_model.pkl")

st.title("Income Prediction")

# Create two number input fields
input1 = st.number_input("Enter Age", min_value=1)  # Assuming age cannot be less than 1
input2 = st.number_input("Enter Experience", min_value=0)  # Assuming experience can't be negative

# Create a button
button_clicked = st.button("Predict Income")

# Check if the button is clicked
if button_clicked:
    user_input = [input1, input2]
    prediction = LR_Model.predict([user_input])
    st.write('Income prediction: $', prediction[0])
