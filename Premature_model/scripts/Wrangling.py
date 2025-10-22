import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from category_encoders import TargetEncoder
#from sklearn.preprocessing import TargetEncoder

from sklearn.preprocessing import TargetEncoder
from sklearn import set_config
# Ensure that sklearn transformers return pandas DataFrames
set_config(transform_output='pandas')

# import pandas as pd


# def categorize_columns(df, key_variables):
#     numeric_cols = []
#     categorical_cols = []
#     dummy_cols = []

#     features = df.columns.to_list()
#     #features.remove(target_variable)
#     features =[item for item in features  if item not in key_variables]

#     # Iterate over each column in DataFrame
#     for col in features:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             # Check if the column is a dummy variable
#             unique_values = pd.Series(df[col].dropna().unique())
            
#             if unique_values.isin([0, 1]).all() and unique_values.size <= 2:
#                 dummy_cols.append(col)
#             else:
#                 numeric_cols.append(col)
#         elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
#             categorical_cols.append(col)
    
#     print(len(numeric_cols)),print(len(categorical_cols)), print(len(dummy_cols)) 
    
#     return numeric_cols, categorical_cols, dummy_cols


def categorize_columns(df, key_variables, category):
    """
    Categorize columns in a DataFrame into numeric, categorical, and dummy types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    key_variables : list, optional
        List of key columns to exclude from classification (e.g., IDs).
    category : list, optional
        List of columns to forcefully classify as categorical.

    Returns
    -------
    numeric_cols : list
        Columns identified as numeric (excluding forced categorical ones).
    categorical_cols : list
        Columns identified as categorical.
    dummy_cols : list
        Columns identified as dummy (binary 0/1).
    """
    
    numeric_cols = []
    categorical_cols = []
    dummy_cols = []

    # Default empty lists if None provided
    key_variables = key_variables or []
    category = category or []

    features = [col for col in df.columns if col not in key_variables]

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Identify dummy (binary) variables
            unique_values = pd.Series(df[col].dropna().unique())
            if unique_values.isin([0, 1]).all() and unique_values.size <= 2:
                dummy_cols.append(col)
            else:
                numeric_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)

    # ✅ Exclude any manually passed categorical columns from numeric list
    numeric_cols = [col for col in numeric_cols if col not in category]

    # ✅ Add manual categories to categorical list if not already present
    categorical_cols = list(set(categorical_cols + category))

    print(f"🔹 Numeric columns: {len(numeric_cols)}")
    print(f"🔹 Categorical columns: {len(categorical_cols)}")
    print(f"🔹 Dummy columns: {len(dummy_cols)}")

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


# def target_encode_dataframe(df, target_col='premature_flag', 
#                             smoothing=0.3, min_samples_leaf=20):
#     """
#     Apply Target Encoding to all categorical columns in a DataFrame.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing categorical and numeric columns.
#     target_col : str, default='premature_flag'
#         Name of the target variable used for encoding.
#     smoothing : float, default=0.3
#         Smoothing factor for target encoding (helps avoid overfitting).
#     min_samples_leaf : int, default=20
#         Minimum samples to take category averages into account.

#     Returns
#     -------
#     df_encoded : pd.DataFrame
#         Transformed DataFrame with categorical variables replaced by target encodings.
#     encoder : category_encoders.TargetEncoder
#         Fitted TargetEncoder instance (useful to transform test data later).
#     """

#     # Copy DataFrame to avoid modifying the original
#     df = df.copy()

#     # Check target column exists
#     if target_col not in df.columns:
#         raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

#     # Separate target
#     y = df[target_col].squeeze()
#     X = df.drop(columns=[target_col])

#     # Select categorical columns automatically
#     cat_cols = X.columns.to_list()

#     if not cat_cols:
#         print("⚠️ No categorical columns found — returning original DataFrame.")
#         return df, None

#     print(f"🔹 Found {len(cat_cols)} categorical columns: {cat_cols}")

#     # Initialize Target Encoder
#     encoder = TargetEncoder(cols=cat_cols, smoothing=smoothing, min_samples_leaf=min_samples_leaf)

#     # Fit and transform
#     X_encoded = encoder.fit_transform(X, y)

#     # Recombine with target
#     df_encoded = pd.concat([X_encoded, y.reset_index(drop=True)], axis=1)

#     print("✅ Target encoding completed successfully.")
#     return df_encoded



def target_encode_dataframe_map(df, target_col='premature_flag', id_col='CASEID'):
    """
    Perform Target Encoding on all categorical columns of the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that includes categorical features, an ID column, and the target column.
    target_col : str
        Name of the target variable.
    id_col : str
        Name of the ID column (will be excluded from encoding).

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with categorical columns replaced by their target encodings.
    encoding_map : dict
        Dictionary mapping each category to its encoded value for each encoded column.
    encoder : TargetEncoder
        Fitted TargetEncoder object.
    """

    # Convert all numeric columns (except ID and target) to string
    cols_to_convert = [c for c in df.columns if c not in ['CASEID', 'premature_flag']]

    df[cols_to_convert] = df[cols_to_convert].astype(str)

    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col, id_col])

    # Detect categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        print("⚠️ No categorical columns found for target encoding.")
        return df, {}, None

    # Initialize Target Encoder
    encoder = TargetEncoder(
        categories='auto',
        target_type='binary',
        smooth='auto',
        cv=5,
        random_state=42
    )

    # Fit & transform the encoder
    X_encoded = encoder.fit_transform(X, y)

    # Create the encoding map (category → encoding)
    encoding_map = {}
    for col, cats, encs in zip(cat_cols, encoder.categories_, encoder.encodings_):
        encoding_map[col] = dict(zip(cats, encs))

    # Rebuild full encoded dataframe (include ID and target)
    df_encoded = pd.concat([df[[id_col]].reset_index(drop=True), X_encoded, y.reset_index(drop=True)], axis=1)

    print(f"✅ Encoded {len(cat_cols)} categorical columns.")
    return df_encoded, encoding_map, encoder



    

