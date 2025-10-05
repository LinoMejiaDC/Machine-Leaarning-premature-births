
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#Sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss
from scipy.stats import ks_2samp

import time 

pd.set_option('display.max_rows', 4000)
pd.set_option('display.precision', 8)

def rank_models(train_data, train_labels, test_data, test_labels, holdout_data, holdout_labels):
    models = {
        'XGBClassifier': xgb.XGBClassifier(scale_pos_weight = 1300/15344),
        'LogisticRegression': LogisticRegression(class_weight='balanced'),
        'LightGBM': lgb.LGBMClassifier(scale_pos_weight = 1300/15344),
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced')
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(train_data, train_labels)
        for data, labels, dataset_type in [(test_data, test_labels, 'Test'), (holdout_data, holdout_labels, 'Holdout')]:
            preds = model.predict(data)
            probs = model.predict_proba(data)[:, 1]
            accuracy = accuracy_score(labels, preds)
            recall = recall_score(labels, preds)
            precision = precision_score(labels, preds)
            f1 = f1_score(labels, preds)
            roc_auc = roc_auc_score(labels, probs)

            #KS
            table_ks= pd.DataFrame({'real_label': labels['SURVEY_TARGET_SSPIVISN40'], 'probs': probs})
            
            positive_probs = table_ks[table_ks['real_label'] == 1]['probs']
            negative_probs = table_ks[table_ks['real_label'] == 0]['probs']
            
            # Calculate the KS statistic
            ks_stat= ks_2samp(positive_probs, negative_probs).statistic

            # label_series = pd.Series(labels)
            # ks_stat = ks_2samp(probs[labels == 0], probs[labels == 1]).statistic

            results.append({
                'Model': name,
                'dataset_type': dataset_type,
                'Accuracy': accuracy,
                'Recall': recall,
                'Precision': precision,
                'F1 Score': f1,
                'KS Stat': ks_stat,
                'ROC AUC': roc_auc
            })
    
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='Model', ascending=False, inplace=True)
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df = results_df.round(4)
    return results_df