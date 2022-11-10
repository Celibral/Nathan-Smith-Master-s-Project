# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def prep_data(df):
    # get average weight gain per Week_1 per check up
    
    mm = df[['Starting_Mass','Starting_Week', 'Week_1', 'Mass_1', 'Week_2', 'Mass_2', 'Week_3', 'Mass_3', 'Week_4', 'Mass_4']]
    mm = mm.astype(int)#.sum(axis=1)
    
    mm['number_of_weight_measurements'] = mm[['Starting_Mass', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']].isin([0]).sum(axis=1)
    
    mm['avg_weight_gain_pw.0'] = None
    mm['avg_weight_gain_pw.1'] = None
    mm['avg_weight_gain_pw.2'] = None
    mm['avg_weight_gain_pw.3'] = None
    
    Week_1s = 0
    change = 0
    avg_change = 0
    
    for index,x in mm.iterrows():
        if (x['Starting_Mass'] != 0) & (x['Mass_1'] != 0):
            Week_1s = x['Week_1'] - x['Starting_Week']
            change = x['Mass_1'] - x['Starting_Mass'] 
            x['avg_weight_gain_pw.0'] = (change/Week_1s)
        else: 
            x['avg_weight_gain_pw.0'] = 0
        
        if (x['Mass_1'] != 0) & (x['Mass_2'] != 0):
            Week_1s = x['Week_3'] - x['Week_1']
            change = x['Mass_2'] - x['Mass_1'] 
    
            x['avg_weight_gain_pw.1'] = (change/Week_1s)
        else: 
            x['avg_weight_gain_pw.1'] = 0
            
        if (x['Mass_2'] != 0) & (x['Mass_3'] != 0):
            Week_1s = x['Week_3'] - x['Week_2']
            change = x['Mass_3'] - x['Mass_2'] 
    
            x['avg_weight_gain_pw.2'] = (change/Week_1s)
        else: 
            x['avg_weight_gain_pw.2'] = 0
            
        if (x['Mass_3'] != 0) & (x['Mass_4'] != 0):
            Week_1s = x['Week_4'] - x['Week_3']
            change = x['Mass_4'] - x['Mass_3'] 
    
            x['avg_weight_gain_pw.3'] = (change/Week_1s)
        else: 
            x['avg_weight_gain_pw.3'] = 0
            
        mm.loc[index] = x
                
    avgs = mm[mm['avg_weight_gain_pw.0'] > 0].mean()
    avgs = avgs[['avg_weight_gain_pw.0', 'avg_weight_gain_pw.1', 'avg_weight_gain_pw.2', 'avg_weight_gain_pw.3']]
    
    df['Starting_Mass'] = np.where(df['Starting_Mass'] == 0, df['Mass_1'] - (avgs['avg_weight_gain_pw.0'] * (df['Week_1'] - df['Starting_Week'])), df['Starting_Mass']) 
    df['Mass_1'] = np.where((df['Mass_2'] != 0) & (df['Mass_1'] == 0), df['Mass_2'] - (avgs['avg_weight_gain_pw.1'] * (df['Week_2'] - df['Week_1'])), np.where(df['Mass_1'] == 0,df['Starting_Mass'] + (avgs['avg_weight_gain_pw.0'] * (df['Week_1'] - df['Starting_Week'])),df['Mass_1'])) 
    df['Mass_2'] = np.where((df['Mass_3'] != 0) & (df['Mass_2'] == 0), df['Mass_3'] - (avgs['avg_weight_gain_pw.2'] * (df['Week_3'] - df['Week_2'])), np.where(df['Mass_2'] == 0,df['Mass_1'] + (avgs['avg_weight_gain_pw.1'] * (df['Week_2'] - df['Week_1'])),df['Mass_2'])) 
    df['Mass_3'] = np.where((df['Mass_4'] != 0) & (df['Mass_3'] == 0), df['Mass_4'] - (avgs['avg_weight_gain_pw.3'] * (df['Week_4'] - df['Week_3'])), np.where(df['Mass_3'] == 0,df['Mass_2'] + (avgs['avg_weight_gain_pw.2'] * (df['Week_3'] - df['Week_2'])),df['Mass_3'])) 
    
    
    # remove patients with no readings for Starting_SBP/DBP
    df = df[df['Starting_SBP'] != 0]

    # remove last checkup readings
    df = df.drop(['Week_4','Mass_4','SBP_4', 'DBP_4', 'Height'], axis=1)
    
    df = encode_rares(df)

    return df

def encode_rares(data):
    
    cat_w_rares = ['Parity']
    
    data[cat_w_rares] = np.where(data[cat_w_rares] > 1, '2', np.where(data[cat_w_rares] == 1, '1', '0'))
    return data


# load raw data into a pandas dataframe

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_1/PE_data.csv', index_col=[0])


#remove unnesasry fields or patients

df = df.drop([34,50,19,70,72,68], axis=0)


# correct value for patient 66: SBP_2 12 -> 120

(df.loc[66, 'SBP_2']) = 120


# correct value for patient 5: DBP_3 774 -> 74

(df.loc[5,'DBP_3']) = 74

df = prep_data(df)

df.to_csv('./Iteration_1/it1_data_preprocessed.csv')