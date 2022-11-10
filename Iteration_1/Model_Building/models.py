import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
# to evaluate the model
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Evaluaation function  ########################################################################################################################################################################################################
def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))
    

df = pd.read_csv('C:/Users/no1ca/OneDrive/Documents/Masters/Final Masters Model/Iteration_1/PE_data.csv', index_col=[0])

y = df['Developed_PE']
df = df.drop(['Developed_PE', 'Diabetic'], axis=1)


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

    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    svm_score = np.add(svm_score, eval(y_test, preds))

    
    # LR #################################################################################################################################

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    LR_score = np.add(LR_score, eval(y_test, preds))
     
    
    # Naive Bayes ########################################################################################################################

    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    NB_score = np.add(NB_score, eval(y_test, preds))
    

    # KNN ###########################################################################################################################
   
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    KNN_score = np.add(KNN_score, eval(y_test, preds))


    # DT ############################################################################################################################
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    DT_score = np.add(DT_score, eval(y_test, preds))
    
    
       
    # RF ###############################################################################################################################
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    RF_score = np.add(RF_score, eval(y_test, preds))
    

for i in range(len(svm_score)):
    if i == 0: 
        print('Accuracy')    
    elif i == 1:
        print('Error Rate')
    elif i == 2:
        print('AUC')
    elif i == 3:
        print('Precision')
    elif i == 4:
        print('Recall')
    elif i == 5:
        print('F1')
    print(str(svm_score[i]/n_splits))
    print(str(LR_score[i]/n_splits))
    print(str(NB_score[i]/n_splits))
    print(str(KNN_score[i]/n_splits))
    print(str(DT_score[i]/n_splits))
    print(str(RF_score[i]/n_splits))
    
