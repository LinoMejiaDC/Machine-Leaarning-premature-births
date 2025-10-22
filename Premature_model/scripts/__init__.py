# scripts/__init__.py

from .utils import (
    get_variable_name,
    count_repetitive_caseids,
    add_prefix_except_caseid,
    get_dummy_variables,
    convert_objects_to_int64_safe,
    drop_null_and_list,
    aggregate_sum_by_caseid,
    analyze_matches,
    analyze_matches_explicit_keys,
    aggregate_with_value_suffix,
    drop_high_null_columns,
    save_file,
    add_value_suffix
)

from .Wrangling import (
    categorize_columns,
    fill_missing_with_mode,
    target_encode, 
    fill_missing_with_median_coding,
    #target_encode_dataframe,
    target_encode_dataframe_map,
    #sklearn_target_encode

    
)

from .modeling import (rank_models,
                       split_dataset, 
                       plot_roc_curve_from_df)

__all__ = [
    # utils.py
    'get_variable_name',
    'count_repetitive_caseids',
    'add_prefix_except_caseid',
    'get_dummy_variables',
    'convert_objects_to_int64_safe',
    'drop_null_and_list',
    'aggregate_sum_by_caseid',
    'analyze_matches',
    'analyze_matches_explicit_keys',
    'aggregate_with_value_suffix',
    'drop_high_null_columns',
    'save_file',
    'add_value_suffix',
    'save_txt',

    # Wrangling.py
    'categorize_columns',
    'fill_missing_with_mode',
    'target_encode', 
    'fill_missing_with_median_coding',
    # 'target_encode_dataframe',
    'target_encode_dataframe_map',
    # 'sklearn_target_encode',

    #modeling
    'rank_models',
    'split_dataset',
    'plot_roc_curve_from_df'
]
