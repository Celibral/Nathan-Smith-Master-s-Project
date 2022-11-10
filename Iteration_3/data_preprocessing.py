# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# load raw data into a pandas dataframe

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_1/it1_data_preprocessed.csv', index_col=[0])


# it2 feature engineering
# combined previous mis and stillbirth


ss1 = df[['HT_History', 'Previous_Miscarriage', 'Previous_Stillbirth',
       'Diabetes_Fhistory', 'Age', 'HT_Fhistory', 'Starting_Week',
       'Starting_Mass', 'Starting_SBP', 'Starting_DBP', 'Week_1',
       'Mass_1', 'SBP_1', 'DBP_1', 'Developed_PE']]

ss2 = df[['Multiple_Fetuses', 'HT_History', 'Previous_Miscarriage',
       'Previous_Stillbirth', 'Diabetic', 'Diabetes_Fhistory', 'Age',
       'HT_Fhistory', 'Starting_Week', 'Starting_Mass', 'Starting_SBP',
       'Starting_DBP', 'Week_1', 'Mass_1', 'SBP_1', 'DBP_1', 'Week_2',
       'Mass_2', 'SBP_2', 'DBP_2', 'Week_3', 'Mass_3', 'SBP_3', 'DBP_3', 'Developed_PE']]

ss3 = df[['Parity', 'HT_History', 'Previous_Miscarriage',
       'Previous_Stillbirth', 'Diabetes_Fhistory', 'Age', 'HT_Fhistory',
       'Starting_Week', 'Starting_Mass', 'Starting_SBP', 'Starting_DBP',
       'Week_1', 'Mass_1', 'SBP_1', 'DBP_1', 'Developed_PE']]


df['Previous_Mis_Still'] = np.where((df['Previous_Stillbirth'] == True) | (df['Previous_Miscarriage'] == True), True, False)

ss4 = df[['HT_History', 'Previous_Mis_Still',
       'Diabetes_Fhistory', 'Age', 'HT_Fhistory', 'Starting_Week',
       'Starting_Mass', 'Starting_SBP', 'Starting_DBP', 'Week_1',
       'Mass_1', 'SBP_1', 'DBP_1', 'Developed_PE']]

ss5 = df[['Multiple_Fetuses', 'HT_History', 'Previous_Mis_Still', 'Diabetic', 'Diabetes_Fhistory', 'Age',
       'HT_Fhistory', 'Starting_Week', 'Starting_Mass', 'Starting_SBP',
       'Starting_DBP', 'Week_1', 'Mass_1', 'SBP_1', 'DBP_1', 'Week_2',
       'Mass_2', 'SBP_2', 'DBP_2', 'Week_3', 'Mass_3', 'SBP_3', 'DBP_3', 'Developed_PE']]

ss6 = df[['Parity', 'HT_History', 'Previous_Mis_Still', 'Diabetes_Fhistory', 'Age', 'HT_Fhistory',
       'Starting_Week', 'Starting_Mass', 'Starting_SBP', 'Starting_DBP',
       'Week_1', 'Mass_1', 'SBP_1', 'DBP_1', 'Developed_PE']]


ss1.to_csv('./Iteration_2/ss1.csv')
ss2.to_csv('./Iteration_2/ss2.csv')
ss3.to_csv('./Iteration_2/ss3.csv')
ss4.to_csv('./Iteration_2/ss4.csv')
ss5.to_csv('./Iteration_2/ss5.csv')
ss6.to_csv('./Iteration_2/ss6.csv')