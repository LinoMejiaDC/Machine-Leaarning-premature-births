
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
#from xgboost import XGBClassifier


from scipy.stats import ks_2samp

import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,  GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


# def split_dataset (df, remove_columns,target_variable='SURVEY_Q8R2'):
    
#     features = df.columns.to_list()
#     features.remove(remove_columns)
#     target_variable= ['SURVEY_Q8R2']
    
#     X = df[features]
#     y = df[target_variable]
    
#     train_data, test_data, train_labels, test_labels = train_test_split( X, y, test_size=0.2222, random_state=123)
#     print (f"train shape, {train_data.shape}, teste shape, {test_data.shape}, train label  shape, {train_labels.shape}, test label train shape, {test_labels.shape}")

#     return train_data, test_data, train_labels, test_labels

from sklearn.model_selection import train_test_split

def split_dataset(df, remove_columns=None, target_variable='premature_flag'):
    """
    Splits a dataset into training and testing sets, ensuring that:
    - The target variable is not included among the features.
    - Any specified columns in remove_columns are also excluded.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and target.
    remove_columns : list or str, optional
        Columns to exclude from modeling (e.g., IDs, names, metadata).
    target_variable : str
        Target column name.
        
    Returns
    -------
    train_data, test_data, train_labels, test_labels, features : tuple
        Train/test splits and list of selected feature names.
    """
    
    # Ensure remove_columns is a list
    if remove_columns is None:
        remove_columns = []
    elif isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    
    # Ensure target_variable is also treated as a list
    target_list = [target_variable] if isinstance(target_variable, str) else target_variable
    
    # Remove both target and unwanted columns from feature set
    features = [col for col in df.columns if col not in remove_columns + target_list]
    
    # Define X (features) and y (target)
    X = df[features]
    y = df[target_variable]
    
    # Split into train/test
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2222, random_state=123, stratify=y
    )
    
    # Print diagnostics
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(f"Train labels shape: {train_labels.shape}, Test labels shape: {test_labels.shape}")
    print(f"Features used: {len(features)}")
    
    return train_data, test_data, train_labels, test_labels, features



# def rank_models(train_data, train_labels, test_data, test_labels, holdout_data, holdout_labels):
#     models = {
#         'XGBClassifier': xgb.XGBClassifier(scale_pos_weight = 1300/15344),
#         'LogisticRegression': LogisticRegression(class_weight='balanced'),
#         'LightGBM': lgb.LGBMClassifier(scale_pos_weight = 1300/15344),
#         'RandomForest': RandomForestClassifier(class_weight='balanced'),
#         'DecisionTree': DecisionTreeClassifier(class_weight='balanced')
#     }
    
#     results = []
    
#     for name, model in models.items():
#         model.fit(train_data, train_labels)
#         for data, labels, dataset_type in [(test_data, test_labels, 'Test'), (holdout_data, holdout_labels, 'Holdout')]:
#             preds = model.predict(data)
#             probs = model.predict_proba(data)[:, 1]
#             accuracy = accuracy_score(labels, preds)
#             recall = recall_score(labels, preds)
#             precision = precision_score(labels, preds)
#             f1 = f1_score(labels, preds)
#             roc_auc = roc_auc_score(labels, probs)

#             #KS
#             table_ks= pd.DataFrame({'real_label': labels['SURVEY_TARGET_SSPIVISN40'], 'probs': probs})
            
#             positive_probs = table_ks[table_ks['real_label'] == 1]['probs']
#             negative_probs = table_ks[table_ks['real_label'] == 0]['probs']
            
#             # Calculate the KS statistic
#             ks_stat= ks_2samp(positive_probs, negative_probs).statistic

#             # label_series = pd.Series(labels)
#             # ks_stat = ks_2samp(probs[labels == 0], probs[labels == 1]).statistic

#             results.append({
#                 'Model': name,
#                 'dataset_type': dataset_type,
#                 'Accuracy': accuracy,
#                 'Recall': recall,
#                 'Precision': precision,
#                 'F1 Score': f1,
#                 'KS Stat': ks_stat,
#                 'ROC AUC': roc_auc
#             })
    
#     results_df = pd.DataFrame(results)
#     results_df.sort_values(by='Model', ascending=False, inplace=True)
#     results_df['Rank'] = range(1, len(results_df) + 1)
#     results_df = results_df.round(4)
#     return results_df


def rank_models(train_data, train_labels, test_data, test_labels, holdout_data, holdout_labels):
    models = {
        'XGBClassifier': xgb.XGBClassifier(),
        'LogisticRegression': LogisticRegression(class_weight='balanced'),
        'LightGBM': lgb.LGBMClassifier(),
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
        'SVM': SVC(probability=True, class_weight='balanced'),
        'GBM': GradientBoostingClassifier()
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

            # KS statistic
            #table_ks = pd.DataFrame({'real_label': labels['premature_flag'], 'probs': probs})
            # Before using labels, flatten it:
            labels = np.ravel(labels)
            table_ks = pd.DataFrame({'real_label': labels, 'probs': probs})
            
            positive_probs = table_ks[table_ks['real_label'] == 1]['probs']
            negative_probs = table_ks[table_ks['real_label'] == 0]['probs']
            ks_stat = ks_2samp(positive_probs, negative_probs).statistic

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


def plot_roc_curve_from_df(model, test_labels, test_data, limit):

    test_probs = model.predict_proba(test_data)[:, 1]
    #test_preds = (test_probs > limit).astype(int)
    test_preds  = np.where(test_probs > limit, 1, 0)

    #df = pd.DataFrame({'true_col': test_labels['premature_flag'].values,'prob_col': test_probs})
    df = pd.DataFrame({'true_col': test_labels.values,'prob_col': test_probs})
 
    # Calculate the ROC curve and AUC
    fpr, tpr, _ = roc_curve(df['true_col'], df['prob_col'])
    roc_auc = auc(fpr, tpr)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot ROC curve on the first subplot
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")

    # Plot Prediction Distribution on the second subplot
    sns.kdeplot(data=df, x='prob_col', hue='true_col', fill=True, common_norm=False, palette="crest", alpha=0.5, ax=ax2)
    ax2.set_title('Prediction Distribution')
    ax2.set_xlabel('Probability of Event')
    ax2.set_ylabel('Density')
    ax2.grid(True)
    ax2.legend(title='test_labels')

    # Display the plot
    plt.tight_layout()
    plt.show()



# def feature_importance(model, train_data, train_labels, test_data, test_labels, top_n):
    
#     start_time = time.time()
    
#     result = permutation_importance(model, test_data, test_labels, n_repeats=30, random_state=123, scoring = 'neg_mean_squared_error')
#     perm_sorted_idx = result.importances_mean.argsort()

#         # Sort the features based on their importance
#     #perm_sorted_idx = result.importances_mean.argsort()[::-1]
    
#     # Create a DataFrame with variable names, their importance, and ranks
#     rank = pd.DataFrame({
#         'variable': test_data.columns[perm_sorted_idx],
#         'value': result.importances_mean[perm_sorted_idx]})
    
#     rank.sort_values(by='value', ascending=False, inplace = True)
#     rank['rank']=np.arange(1, len(perm_sorted_idx) + 1)
    
#     #top_features_indices = perm_sorted_idx[:2]  # Indices of the top 2 features
#     top_features_indices = perm_sorted_idx[-top_n:] 

#     # Plotting only the top 2 features
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12))
#     ax1.barh(np.arange(top_n), result.importances_mean[top_features_indices], color='skyblue')
#     ax1.set_yticks(np.arange(top_n))
#     ax1.set_yticklabels(test_data.columns[top_features_indices])
#     ax1.set_title('Top 2 Feature Importances')
#     ax1.set_xlabel('Importance')

#     ax2.boxplot(result.importances[top_features_indices].T, vert=False, labels=test_data.columns[top_features_indices])
#     ax2.set_title('Permutation Importances of Top 2 Features')
#     fig.tight_layout()

#     end_time = time.time()  # Record the end time
#     elapsed_time = (end_time - start_time) / 60  # Calculate elapsed time in minutes
#     print(f"Processing time: {elapsed_time:.2f} minutes")  # Print the processing time
    
#     return rank , plt.show()


def feature_importance(model, test_data, test_labels, top_n=10, scoring='roc_auc', n_repeats=10):
    """
    Compute and visualize permutation feature importance for a trained model.
    Works for XGB, LightGBM, RandomForest, etc.
    
    Parameters
    ----------
    model : fitted model
        The trained classifier.
    test_data : pd.DataFrame
        Test features.
    test_labels : pd.Series or np.array
        True labels.
    top_n : int
        Number of top features to plot.
    scoring : str
        Metric to use for importance ('roc_auc', 'accuracy', 'f1', etc.)
    n_repeats : int
        Number of shuffling repetitions.
    """
    start_time = time.time()

    # Run permutation importance
    result = permutation_importance(
        model, test_data, test_labels,
        n_repeats=n_repeats,
        random_state=123,
        scoring=scoring,
        n_jobs=-1
    )

    # Sort descending
    sorted_idx = result.importances_mean.argsort()[::-1]

    # Create ranking dataframe
    rank = pd.DataFrame({
        'variable': test_data.columns[sorted_idx],
        'importance_mean': result.importances_mean[sorted_idx],
        'importance_std': result.importances_std[sorted_idx]
    })
    rank['rank'] = np.arange(1, len(rank) + 1)

    # Plot top N
    top_features = rank.head(top_n)
    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 2))
    ax.barh(top_features['variable'][::-1], top_features['importance_mean'][::-1], color='skyblue')
    ax.set_title(f'Top {top_n} Feature Importances (Permutation)')
    ax.set_xlabel('Mean Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()

    elapsed_time = (time.time() - start_time) / 60
    print(f"⏱ Processing time: {elapsed_time:.2f} minutes")

    return rank,plt.show()


def plot_confusion_matrix(model, X, y, dataset_name='Test'):
    """
    Compute and plot a confusion matrix for a trained classification model.
    
    Parameters
    ----------
    model : fitted classifier
        Your trained model (e.g., XGBClassifier, LogisticRegression, etc.)
    X : pd.DataFrame or np.array
        Features to predict on.
    y : pd.Series or np.array
        True labels.
    dataset_name : str
        Label for plot title (e.g., 'Test', 'Holdout').
    """
    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # Compute confusion matrix
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix ({dataset_name} Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    return plt.show()


def train_perm_importance_gb(
    df, target,
    test_size=0.3,
    random_state=42,
    scoring="roc_auc",
    n_repeats=10
):
    """
    Fast, self-contained permutation-importance workflow using a light GradientBoostingClassifier.
    - Uses ONLY numeric features (keeps it quick; avoids encoding).
    - Median-imputes missing numerics.
    - Trains a small GB model.
    - Computes permutation importance on the held-out test set.

    Returns:
        df_imp  : DataFrame [rank, feature, importance_mean, importance_std]
        result  : sklearn permutation_importance result (for advanced plotting)
        pipe    : fitted sklearn Pipeline (imputer + model)
        X_test  : test features (DataFrame, post column selection)
        y_test  : test targets (Series)
    """
    # Split features / target
    y = df[target]
    X = df.drop(columns=[target])

    # Keep only numeric columns for speed & simplicity
    num_cols = X.select_dtypes(include=[np.number]).columns
    X = X[num_cols].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() == 2 else None
    )

    # Lightweight Gradient Boosting model
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=120,        # small, quick
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=random_state
        ))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Permutation importance on held-out test set
    result = permutation_importance(
        pipe, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=None
    )

    # Rank table
    df_imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)
    df_imp = df_imp[["rank", "feature", "importance_mean", "importance_std"]]

    return df_imp, result


