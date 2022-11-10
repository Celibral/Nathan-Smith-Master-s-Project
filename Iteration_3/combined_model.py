import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
# to evaluate the model
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# Evaluaation function  ########################################################################################################################################################################################################
def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))
    
ss2 = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_2/SS1.csv', index_col=[0])

ss1 = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_2/it1_data_preprocessed.csv', index_col=[0])


ss1 = ss1.drop(['Diabetic', 'Week_1', 'Week_2', 'Week_3', 'Multiple_Fetuses', 'Mass_1', 'Mass_2', 'Mass_3', 'Previous_Stillbirth', 'Previous_Miscarriage', 'Diabetes_Fhistory'], axis=1)
ss2 = ss2.drop(['Week_1', 'Mass_1', 'Previous_Stillbirth', 'Previous_Miscarriage', 'Diabetic', 'Diabetes_Fhistory'], axis=1)

datasets = [ss2]

scores = pd.DataFrame(columns = ['Accuracy','Error Rate','ROC AUC','Precision','Recall','F1 Score'])

j = 1

for i in datasets:
    df = i
    y = df['Developed_PE']
    df = df.drop('Developed_PE', axis=1)
    # initialise cross-validation

    n_splits = 60

    kf = KFold(n_splits=n_splits, shuffle=False)

    cv = RepeatedStratifiedKFold(n_splits=(10), n_repeats=6, random_state=1)


    # load scaler

    scaler = joblib.load('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/minmax_scaler.gz')


    # arrays to store scores from CV

    svm_score = [0, 0, 0, 0, 0, 0]
    LR_score = [0, 0, 0, 0, 0, 0]
    NB_score = [0, 0, 0, 0, 0, 0]
    SGDC_score = [0, 0, 0, 0, 0, 0]
    KNN_score = [0, 0, 0, 0, 0, 0]
    DT_score = [0, 0, 0, 0, 0, 0]
    RF_score = [0, 0, 0, 0, 0, 0]
    GBC_score = [0, 0, 0, 0, 0, 0]
    Ada_score = [0, 0, 0, 0, 0, 0]
    comb_score = [0, 0, 0, 0, 0, 0]
    # loop and split for each split

    for train_index, test_index in cv.split(df, y):
        
        X_train =  df.iloc[train_index]
        X_test =  df.iloc[test_index]
        y_train  =  y.iloc[train_index]
        y_test =  y.iloc[test_index]
        

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

        model = SVC(random_state=0, C = 1)
        model.fit(X_train, y_train)
        svm_preds = model.predict(X_test)
        svm_score = np.add(svm_score, eval(y_test, svm_preds))

        
        # LR #################################################################################################################################

        model = LogisticRegression(random_state=0)
        model.fit(X_train, y_train)
        lr_preds = model.predict(X_test)
        LR_score = np.add(LR_score, eval(y_test, lr_preds))
        
        
        # Naive Bayes ########################################################################################################################

        model = GaussianNB()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        NB_score = np.add(NB_score, eval(y_test, preds))
        

        # KNN ###########################################################################################################################

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        knn_preds = model.predict(X_test)
        KNN_score = np.add(KNN_score, eval(y_test, knn_preds))


        # DT ############################################################################################################################
        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        DT_score = np.add(DT_score, eval(y_test, preds))
        
        
        
        # RF ###############################################################################################################################
        model = RandomForestClassifier(random_state=0)
        model.fit(X_train, y_train)
        rf_preds = model.predict(X_test)
        RF_score = np.add(RF_score, eval(y_test, rf_preds))
        

        # SGDC ##############################################################################################################################

        model = SGDClassifier(random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        SGDC_score = np.add(SGDC_score, eval(y_test, preds))

        # GBC #################################################################################################################################

        model = GradientBoostingClassifier(random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        GBC_score = np.add(GBC_score, eval(y_test, preds))
        

        # AdaBoost 
        from sklearn.ensemble import AdaBoostClassifier
        
        model = AdaBoostClassifier(random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        Ada_score = np.add(Ada_score, eval(y_test, preds))

        comb_preds = []
        for i in range(len(lr_preds)):
            x = ((lr_preds[i]) + (svm_preds[i]) + 2*knn_preds[i])/4
            comb_preds.append(round(x))

        comb_score = np.add(comb_score, eval(y_test, comb_preds))

    s1 = {'Model': 'SVM', 'SS': j, 'Accuracy': (svm_score[0]/n_splits), 'Error Rate': (svm_score[1]/n_splits), 'ROC AUC': (svm_score[2]/n_splits), 'Precision': (svm_score[3]/n_splits), 'Recall': (svm_score[4]/n_splits), ' F1 Score': (svm_score[5]/n_splits)}
    s2 = {'Model': 'LR', 'SS': j, 'Accuracy': (LR_score[0]/n_splits), 'Error Rate': (LR_score[1]/n_splits), 'ROC AUC': (LR_score[2]/n_splits), 'Precision': (LR_score[3]/n_splits), 'Recall': (LR_score[4]/n_splits), ' F1 Score': (LR_score[5]/n_splits)}
    s3 = {'Model': 'NB', 'SS': j, 'Accuracy': (NB_score[0]/n_splits), 'Error Rate': (NB_score[1]/n_splits), 'ROC AUC': (NB_score[2]/n_splits), 'Precision': (NB_score[3]/n_splits), 'Recall': (NB_score[4]/n_splits), ' F1 Score': (NB_score[5]/n_splits)}
    s4 = {'Model': 'KNN', 'SS': j, 'Accuracy': (KNN_score[0]/n_splits), 'Error Rate': (KNN_score[1]/n_splits), 'ROC AUC': (KNN_score[2]/n_splits), 'Precision': (KNN_score[3]/n_splits), 'Recall': (KNN_score[4]/n_splits), ' F1 Score': (KNN_score[5]/n_splits)}
    s5 = {'Model': 'DT', 'SS': j, 'Accuracy': (DT_score[0]/n_splits), 'Error Rate': (DT_score[1]/n_splits), 'ROC AUC': (DT_score[2]/n_splits), 'Precision': (DT_score[3]/n_splits), 'Recall': (DT_score[4]/n_splits), ' F1 Score': (DT_score[5]/n_splits)}
    s6 = {'Model': 'RF', 'SS': j, 'Accuracy': (RF_score[0]/n_splits), 'Error Rate': (RF_score[1]/n_splits), 'ROC AUC': (RF_score[2]/n_splits), 'Precision': (RF_score[3]/n_splits), 'Recall': (RF_score[4]/n_splits), ' F1 Score': (RF_score[5]/n_splits)}

    s7 = {'Model': 'SGBC','SS': j,'Accuracy': (SGDC_score[0]/n_splits), 'Error Rate': (SGDC_score[1]/n_splits), 'ROC AUC': (SGDC_score[2]/n_splits), 'Precision': (SGDC_score[3]/n_splits), 'Recall': (SGDC_score[4]/n_splits), ' F1 Score': (SGDC_score[5]/n_splits)}
    s8 = {'Model': 'GBC','SS': j,'Accuracy': (GBC_score[0]/n_splits), 'Error Rate': (GBC_score[1]/n_splits), 'ROC AUC': (GBC_score[2]/n_splits), 'Precision': (GBC_score[3]/n_splits), 'Recall': (GBC_score[4]/n_splits), ' F1 Score': (GBC_score[5]/n_splits)}
    s9 = {'Model': 'Ada','SS': j,'Accuracy': (Ada_score[0]/n_splits), 'Error Rate': (Ada_score[1]/n_splits), 'ROC AUC': (Ada_score[2]/n_splits), 'Precision': (Ada_score[3]/n_splits), 'Recall': (Ada_score[4]/n_splits), ' F1 Score': (Ada_score[5]/n_splits)}

    scores = scores.append([s1,s2,s3,s4,s5,s6,s7, s8, s9], ignore_index=True)
    s10 = {'Model': 'Combined','SS': j,'Accuracy': (comb_score[0]/n_splits), 'Error Rate': (comb_score[1]/n_splits), 'ROC AUC': (comb_score[2]/n_splits), 'Precision': (comb_score[3]/n_splits), 'Recall': (comb_score[4]/n_splits), ' F1 Score': (comb_score[5]/n_splits)}
    scores = scores.append([s10], ignore_index=True)

    j += 1
    


scores.to_csv('./Iteration_3/test5.csv')
    

        
