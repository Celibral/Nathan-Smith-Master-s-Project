# -*- coding: utf-8 -*-

import pandas as pd
import math
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
# to evaluate the model
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
import joblib
# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

def prep_data(df):
    # get average weight gain per Week_1 per check up
    
    mm = df[['Starting_Weight','Starting_Week', 'Week_1', 'Mass_1', 'Week_2', 'Mass_2', 'Week_3', 'Mass_3', 'Week_4', 'Mass_4']]
    mm = mm.astype(int)#.sum(axis=1)
    
    mm['number_of_weight_measurements'] = mm[['Starting_Weight', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']].isin([0]).sum(axis=1)
    
    mm['avg_weight_gain_pw.0'] = None
    mm['avg_weight_gain_pw.1'] = None
    mm['avg_weight_gain_pw.2'] = None
    mm['avg_weight_gain_pw.3'] = None
    
    Week_1s = 0
    change = 0
    avg_change = 0
    
    for index,x in mm.iterrows():
        if (x['Starting_Weight'] != 0) & (x['Mass_1'] != 0):
            Week_1s = x['Week_1'] - x['Starting_Week']
            change = x['Mass_1'] - x['Starting_Weight'] 
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
    
    df['Starting_Weight'] = np.where(df['Starting_Weight'] == 0, df['Mass_1'] - (avgs['avg_weight_gain_pw.0'] * (df['Week_1'] - df['Starting_Week'])), df['Starting_Weight']) 
    df['Mass_1'] = np.where((df['Mass_2'] != 0) & (df['Mass_1'] == 0), df['Mass_2'] - (avgs['avg_weight_gain_pw.1'] * (df['Week_2'] - df['Week_1'])), np.where(df['Mass_1'] == 0,df['Starting_Weight'] + (avgs['avg_weight_gain_pw.0'] * (df['Week_1'] - df['Starting_Week'])),df['Mass_1'])) 
    df['Mass_2'] = np.where((df['Mass_3'] != 0) & (df['Mass_2'] == 0), df['Mass_3'] - (avgs['avg_weight_gain_pw.2'] * (df['Week_3'] - df['Week_2'])), np.where(df['Mass_2'] == 0,df['Mass_1'] + (avgs['avg_weight_gain_pw.1'] * (df['Week_2'] - df['Week_1'])),df['Mass_2'])) 
    df['Mass_3'] = np.where((df['Mass_4'] != 0) & (df['Mass_3'] == 0), df['Mass_4'] - (avgs['avg_weight_gain_pw.3'] * (df['Week_4'] - df['Week_3'])), np.where(df['Mass_3'] == 0,df['Mass_2'] + (avgs['avg_weight_gain_pw.2'] * (df['Week_3'] - df['Week_2'])),df['Mass_3'])) 
    
    
    df = df.drop(['Week_4','Mass_4'], axis=1)
    df = df.drop(['Week_2', 'Mass_2', 'Week_3', 'Mass_3','SBP_2', 'DBP_2', 'SBP_3', 'DBP_3'], axis=1)
    
    '''
    for index,x in df.iterrows():
       if (x['Starting_SBP] == 0) & (x['SBP_1] != 0):
           x['Starting_SBP] = x['SBP_1]
           x['Starting_DBP'] = x['DBP_1']
       elif (x['Starting_SBP] == 0) & (x['SBP_1] == 0):
           x['Starting_SBP] = x['SBP_2']
           x['Starting_DBP'] = x['DBP_2']
       elif (x['SBP_1] == 0):
           x['SBP_1] = x['SBP_2']
           x['DBP_1'] = x['DBP_2']
           
       df.loc[index] = x
'''
    
    return df

def encode_rares(data):
    
    cat_w_rares = ['Parity']
    
    for var in cat_w_rares:
        
        # find the frequent categories
        frequent_ls = find_frequent_labels(data, var, 0.25)
        
        # replace rare categories by the string "Rare"
        data[var] = np.where(data[var].isin(
            frequent_ls), data[var], '3')
        
        return data
    
def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    tmp = df.groupby(var)[var].count() / len(df)

    return tmp[tmp > rare_perc].index

# Evaluaation function  ########################################################################################################################################################################################################

def eval(y_test, preds):
    print(f'Accuracy: ' + str(accuracy_score(y_test, preds)))
    print(f'Error Rate: ' + str(1 - (accuracy_score(y_test, preds))))
    print(f'ROC_AUC Score:' + str(roc_auc_score(y_test, preds)))
    print(f'Confusion Matrix: ')
    print(str(confusion_matrix(y_test, preds)))
    print(f'Classification Report: ')
    print(str(classification_report(y_test, preds)))
    
    import matplotlib.pyplot as plt
    
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([False, True], [False, True],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/PE_data.csv', index_col=[0])


#remove unnesasry fields or patients

df = df.drop(['Unnamed: 0','SBP_4', 'DBP_4', 'Height'], axis=1)
df = df.drop([34,50,19,70,72,68], axis=0)


# correct value for patient 66: SBP_2 12 -> 120

(df.loc[66, 'SBP_2']) = 120


# correct value for patient 5: DBP_3 774 -> 74

(df.loc[5,'DBP_3']) = 74

df = prep_data(df)

df.to_csv('iteration_1_data.csv')
'''
# ensure integer features are integers

df[['Starting_Weight', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']] = df[['Starting_Weight', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']].astype(int) 

y = df['Developed PE']
df = df.drop('Developed PE', axis=1)


# initialise cross-validation

n_splits = 60

kf = KFold(n_splits=n_splits, shuffle=False)

cv = RepeatedStratifiedKFold(n_splits=(10), n_repeats=6, random_state=1)


# load scaler
scaler = joblib.load('C:/Users/no1ca/OneDrive/Documents/Masters/masters_PE_model/minmax_scaler.joblib')


# loop and split for each split
    
for train_index, test_index in cv.split(df, y):
    
    X_train =  df.iloc[train_index]
    X_test =  df.iloc[test_index]
    y_train  =  y.iloc[train_index]
    y_test =  y.iloc[test_index]

    
    # replace missing values
    
    X_train = prep_data(X_train)
    X_test = prep_data(X_test)
    
    X_train = X_train[set4]
    X_test = X_test[set4]
    

    # scale features using min and max    
    #  fit  the scaler to the train set

    scaler.fit(X_train) 
    

    # transform the train and test set
    
    X_train = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns
    )
    
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns
    )
    
    
    # Models ##############################################################################################################################
    

    # SVM #################################################################################################################################

    model = SVC(random_state=0, C=10,kernel='linear')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    svm_score = np.add(svm_score, eval(y_test, preds))

    
    # LR #################################################################################################################################

    model = LogisticRegression(random_state=0, C=2, penalty='l2')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    LR_score = np.add(LR_score, eval(y_test, preds))
     
    
    # Naive Bayes ########################################################################################################################

    model = GaussianNB(var_smoothing=0.01)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    NB_score = np.add(NB_score, eval(y_test, preds))
    
    
    # SGDC ##############################################################################################################################

    model = SGDClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    SGDC_score = np.add(SGDC_score, eval(y_test, preds))

    # KNN ###########################################################################################################################
   
    model = KNeighborsClassifier(leaf_size=1, n_neighbors=6, p=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    KNN_score = np.add(KNN_score, eval(y_test, preds))


    # DT ############################################################################################################################
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    DT_score = np.add(DT_score, eval(y_test, preds))
    
    
       
    # RF ###############################################################################################################################
    model = RandomForestClassifier(random_state=(0), bootstrap=(True), max_features = 'auto', min_samples_leaf = 4, min_samples_split = 2, n_estimators = 5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    RF_score = np.add(RF_score, eval(y_test, preds))
    

    # GBC #################################################################################################################################
  
    model = GradientBoostingClassifier(random_state=0, max_depth=1, max_features=('sqrt'), min_samples_leaf=1, min_samples_split=2, n_estimators=7)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    GBC_score = np.add(GBC_score, eval(y_test, preds))
    

    # AdaBoost 
    from numpy import mean
    from numpy import std
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import AdaBoostClassifier
    
    model = AdaBoostClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    model = AdaBoostClassifier(n_estimators=20, learning_rate=(0.01),random_state=0, algorithm='SAMME')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    Ada_score = np.add(Ada_score, eval(y_test, preds))
     
    # Iteration Three  ########################################################################################################################################################################################################
    ########################################################################################################################################################################################################
    
    # Remove Outliers ########################################################################################################################################################################################################
    
    model1 = SVC(random_state=0)
    model1.fit(X_train, y_train)
    
    model2 = LogisticRegression(random_state=0)
    model2.fit(X_train, y_train)
    
    model3 = model = KNeighborsClassifier()
    model3.fit(X_train, y_train)
    
    #preds = model1.predict(X_test)
    #eval(y_test, preds)
    
    from sklearn.ensemble import VotingClassifier
    
    est_Ensemble = VotingClassifier(estimators=[('SVM', model1), ('LR', model2), ('KNN', model3)],
                            weights=[1,1,1])
    
    score_Ensemble=est_Ensemble.fit(X_train,y_train).score(X_test,y_test)
    comb_score = np.add(comb_score, eval(y_test, est_Ensemble.predict(X_test)))
 
'''