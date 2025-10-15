import pandas as pd
import numpy as np
import os

def counter(a,b):
    c  = a +b
    return c


def get_variable_name(df):
    for name, value in globals().items():
        if value is df:
            return name


def count_repetitive_caseids(df):
    if 'CASEID' in df.columns:  # Check if 'CASEID' column exists
        duplicate_count = df['CASEID'].duplicated().sum()  # Count duplicates
        total_count = df['CASEID'].count()  # Count total rows
        unique_count = df['CASEID'].nunique()  # Count unique CASEID
        return duplicate_count, total_count, unique_count
    else:
        return None, None, None  # If 'CASEID' column is missing
    


def add_prefix_except_caseid(df, prefix):
    """
    Add a prefix to all columns in the dataframe except 'CASEID'.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        prefix (str): The prefix to add.

    Returns:
        pd.DataFrame: A new dataframe with prefixed column names.
    """
    df = df.copy()
    new_columns = {
        col: f"{prefix}_{col}" if col != 'CASEID' else col
        for col in df.columns
    }
    return df.rename(columns=new_columns)

def get_dummy_variables(df):
    """
    Identify dummy (binary) variables in a DataFrame.
    A dummy variable is defined as having only 0, 1, or NaN values.
    Columns that are entirely NaN are excluded.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[str]: List of column names that are dummy variables.
    """
    dummy_cols = []
    for col in df.columns:
        # Skip if all values are NaN
        if df[col].isna().all():
            continue
        
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            dummy_cols.append(col)
    return dummy_cols



def convert_objects_to_int64_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns in a DataFrame to Int64 if possible.
    If conversion is not possible (non-integer strings), keep the column as object.
    """
    df_converted = df.copy()

    for col in df_converted.select_dtypes(include=["object"]).columns:
        s = df_converted[col].astype(str).str.strip()   # remove extra spaces
        s = s.mask(s == "", np.nan)                     # empty string -> NaN (no downcasting warning)
        
        # Try converting
        try:
            converted = pd.to_numeric(s, errors="raise").astype("Int64")
            df_converted[col] = converted
        except Exception:
            df_converted[col] = df_converted[col]  # keep original if fails
    
    return df_converted


def drop_null_and_list(df, exclude_list):
    """
    Drop columns that are entirely NaN or present in the exclude_list.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        exclude_list (list): List of column names to exclude.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    # Drop all-null columns
    df_filtered = df.dropna(axis=1, how="all")
    
    # Drop columns from exclude_list
    df_filtered = df_filtered.drop(columns=[col for col in exclude_list if col in df_filtered.columns], errors="ignore")
    
    return df_filtered

def aggregate_sum_by_caseid(df, caseid_col="CASEID"):
    """
    Aggregate a DataFrame at the CASEID level by summing numeric columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        caseid_col (str): Name of the column that identifies the case/group.

    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per CASEID.
    """
    # Group by CASEID and sum only numeric columns
    aggregated_df = (
        df.groupby(caseid_col, as_index=False)
          .sum(numeric_only=True)
    )
    return aggregated_df


# def aggregate_with_value_suffix(df, caseid_col="CASEID", exclude_cols=None):
#     """
#     Aggregate at CASEID level by creating dummy columns for each unique value in selected columns,
#     and summing the counts. Excludes columns specified in exclude_cols.

#     Parameters:
#         df (pd.DataFrame): Input DataFrame
#         caseid_col (str): Column name to group by
#         exclude_cols (list): List of columns to exclude from processing

#     Returns:
#         pd.DataFrame: Aggregated DataFrame with CASEID and dummy count columns
#     """
#     if exclude_cols is None:
#         exclude_cols = []

#     # Columns to process: all except caseid_col and excluded ones
#     cols_to_process = [c for c in df.columns if c not in [caseid_col] + exclude_cols]

#     # Create dummy variables for the selected columns
#     df_dummies = pd.get_dummies(df[cols_to_process].astype(str), prefix=cols_to_process)

#     # Combine CASEID with the dummies
#     df_combined = pd.concat([df[[caseid_col]], df_dummies], axis=1)

#     # Aggregate by CASEID summing the dummy counts
#     #agg_df = df_combined.groupby(caseid_col, as_index=False).sum()

#     return df_combined

def aggregate_with_value_suffix(df, caseid_col="CASEID", exclude_cols=None):
    """
    Aggregate at CASEID level by creating dummy columns for each unique value in selected columns,
    and summing the counts. Excludes columns specified in exclude_cols.
    """
    if exclude_cols is None:
        exclude_cols = []

    cols_to_process = [c for c in df.columns if c not in [caseid_col] + exclude_cols]

    # Create dummy variables as numeric (0/1)
    df_dummies = pd.get_dummies(df[cols_to_process].astype(str), prefix=cols_to_process).astype(int)

    # Combine CASEID with the dummies
    df_combined = pd.concat([df[[caseid_col]], df_dummies], axis=1)

    return df_combined


def analyze_matches(left_df, right_df, label):
    original_rows = len(left_df)
    right_rows = len(right_df)
    
    merged = left_df.merge(right_df, on='CASEID', how='left', indicator=True)
    merged_rows = len(merged)

    match_count = (merged['_merge'] == 'both').sum()
    match_pct = match_count / original_rows * 100

    print(f"🔎 {label}")
    print(f" - Rows in target_final: {original_rows}")
    print(f" - Rows in right dataframe ({label}): {right_rows}")
    print(f" - Rows after merge: {merged_rows}")
    print(f" ✅ Matches on CASEID: {match_count} ({match_pct:.2f}%)\n")


def analyze_matches_explicit_keys(target_df, right_df, label, right_key):
    # Drop duplicates in right dataframe based on CASEID and the specific key
    right_df_clean = right_df.drop_duplicates(subset=['CASEID', right_key])

    # Perform merge using explicit keys
    merged = target_df.merge(
        right_df_clean,
        left_on=['CASEID', 'BIDX'],
        right_on=['CASEID', right_key],
        how='left',
        indicator=True
    )

    # Count and summarize
    original_rows = len(target_df)
    right_rows = len(right_df)
    merged_rows = len(merged)
    match_count = (merged['_merge'] == 'both').sum()
    match_pct = match_count / original_rows * 100

    # Print results
    print(f"🔎 {label}")
    print(f" - Rows in target_final: {original_rows}")
    print(f" - Rows in right dataframe: {right_rows}")
    print(f" - Matches on CASEID + {right_key}: {match_count} ({match_pct:.2f}%)\n")


def save_file(df, dir = "data\\interim", output_file = "Modulo1633_REC41_2024_fil_clear.csv"):

    # File name only

    # Go one level up from the current working directory
    base_dir = os.path.dirname(os.getcwd())   # gives "c:\\Users\\linoc\\OneDrive\\Encoder\\03_partos"
    output_dir = os.path.join(base_dir, dir)

    #Full path
    output_path = os.path.join(output_dir, output_file)

    # Save DataFrame
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def drop_high_null_columns(df, threshold=0.45):
    """
    Drop columns from a DataFrame where the percentage of null values 
    is greater than the given threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Proportion threshold (default 0.45 means 45%).

    Returns:
        pd.DataFrame: DataFrame with high-null columns removed.
    """
    # Calculate the null ratio for each column
    null_ratio = df.isnull().mean()

    # Keep only columns with null ratio <= threshold
    df_filtered = df.loc[:, null_ratio <= threshold]

    return df_filtered


def add_value_suffix(df):
    """
    Convert categorical values into dummy columns (0/1 encoding) for each unique value 
    in the dataframe columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with dummy variables (0/1 instead of True/False)
    """
    # Columns to process (all columns from df)
    cols_to_process = df.columns

    # Create dummy variables with dtype=int to ensure 0/1 instead of True/False
    df_dummies = pd.get_dummies(
        df[cols_to_process].astype(str),
        prefix=cols_to_process,
        dtype=int
    )

    return df_dummies



