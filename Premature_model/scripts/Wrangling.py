import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def categorize_columns(df, key_variables):
    numeric_cols = []
    categorical_cols = []
    dummy_cols = []

    features = df.columns.to_list()
    #features.remove(target_variable)
    features =[item for item in features  if item not in key_variables]

    # Iterate over each column in DataFrame
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if the column is a dummy variable
            unique_values = pd.Series(df[col].dropna().unique())
            
            if unique_values.isin([0, 1]).all() and unique_values.size <= 2:
                dummy_cols.append(col)
            else:
                numeric_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)
    
    print(len(numeric_cols)),print(len(categorical_cols)), print(len(dummy_cols)) 
    
    return numeric_cols, categorical_cols, dummy_cols


# function missing values 
def fill_missing_with_mode(df):
    """
    Fills missing values in each column of the DataFrame with the median of that column.

    Parameters:
    df (pd.DataFrame): The DataFrame with missing values.

    Returns:
    pd.DataFrame: A DataFrame with missing values filled with the median of their respective columns.
    """
    df_filled = df.copy()
    
    for column in df_filled.columns:
        mode_value = df_filled[column].mode()[0]
        #df_filled[column].fillna(median_value, inplace=True)
        df_filled[column] = df_filled[column].fillna(mode_value)
    
    return df_filled


# function encoding 
def target_encode(df, categorical_columns, target_column):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_encoded = df.copy()
    
    # For each categorical feature, perform target encoding
    for column in categorical_columns:
        # Create a dictionary of category: average target
        target_means = df.groupby(column)[target_column].mean()
        # Map the categorical features to these target averages
        df_encoded[column] = df[column].map(target_means)
    
    df_encoded.drop(columns = target_column, inplace = True)
        
    return df_encoded


def fill_missing_with_median_coding(df):
    """
    1. Cap values above 99th percentile to the max below that threshold.
    2. Fill missing values with column median.
    3. Scale each column between 0 and 1 with MinMaxScaler.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric columns only.

    Returns:
    -------
    pd.DataFrame
        Transformed DataFrame with capped, imputed, and scaled values.
    """
    df_filled = df.copy()

    for column in df_filled.columns:
        # Step 1: cap at 99th percentile
        percentile_99 = df_filled[column].quantile(0.99)
        max_value_below_99 = df_filled.loc[df_filled[column] < percentile_99, column].max()
        df_filled.loc[df_filled[column] > percentile_99, column] = max_value_below_99

        # Step 2: fill missing with median
        median_value = df_filled[column].median()
        df_filled[column] = df_filled[column].fillna(median_value)

        # Step 3: scale between 0 and 1
        scaler = MinMaxScaler()
        df_filled[[column]] = scaler.fit_transform(df_filled[[column]])

    print("Shape after transformation:", df_filled.shape)
    return df_filled