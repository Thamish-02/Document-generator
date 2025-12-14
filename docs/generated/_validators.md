## AI Summary

A file named _validators.py.


## Class: Remediation

### Function: num_examples_validator(df)

**Description:** This validator will only print out the number of examples and recommend to the user to increase the number of examples if less than 100.

### Function: necessary_column_validator(df, necessary_column)

**Description:** This validator will ensure that the necessary column is present in the dataframe.

### Function: additional_column_validator(df, fields)

**Description:** This validator will remove additional columns from the dataframe.

### Function: non_empty_field_validator(df, field)

**Description:** This validator will ensure that no completion is empty.

### Function: duplicated_rows_validator(df, fields)

**Description:** This validator will suggest to the user to remove duplicate rows if they exist.

### Function: long_examples_validator(df)

**Description:** This validator will suggest to the user to remove examples that are too long.

### Function: common_prompt_suffix_validator(df)

**Description:** This validator will suggest to add a common suffix to the prompt if one doesn't already exist in case of classification or conditional generation.

### Function: common_prompt_prefix_validator(df)

**Description:** This validator will suggest to remove a common prefix from the prompt if a long one exist.

### Function: common_completion_prefix_validator(df)

**Description:** This validator will suggest to remove a common prefix from the completion if a long one exist.

### Function: common_completion_suffix_validator(df)

**Description:** This validator will suggest to add a common suffix to the completion if one doesn't already exist in case of classification or conditional generation.

### Function: completions_space_start_validator(df)

**Description:** This validator will suggest to add a space at the start of the completion if it doesn't already exist. This helps with tokenization.

### Function: lower_case_validator(df, column)

**Description:** This validator will suggest to lowercase the column values, if more than a third of letters are uppercase.

### Function: read_any_format(fname, fields)

**Description:** This function will read a file saved in .csv, .json, .txt, .xlsx or .tsv format using pandas.
 - for .xlsx it will read the first sheet
 - for .txt it will assume completions and split on newline

### Function: format_inferrer_validator(df)

**Description:** This validator will infer the likely fine-tuning format of the data, and display it to the user if it is classification.
It will also suggest to use ada and explain train/validation split benefits.

### Function: apply_necessary_remediation(df, remediation)

**Description:** This function will apply a necessary remediation to a dataframe, or print an error message if one exists.

### Function: accept_suggestion(input_text, auto_accept)

### Function: apply_optional_remediation(df, remediation, auto_accept)

**Description:** This function will apply an optional remediation to a dataframe, based on the user input.

### Function: estimate_fine_tuning_time(df)

**Description:** Estimate the time it'll take to fine-tune the dataset

### Function: get_outfnames(fname, split)

### Function: get_classification_hyperparams(df)

### Function: write_out_file(df, fname, any_remediations, auto_accept)

**Description:** This function will write out a dataframe to a file, if the user would like to proceed, and also offer a fine-tuning command with the newly created file.
For classification it will optionally ask the user if they would like to split the data into train/valid files, and modify the suggested command to include the valid set.

### Function: infer_task_type(df)

**Description:** Infer the likely fine-tuning task type from the data

### Function: get_common_xfix(series, xfix)

**Description:** Finds the longest common suffix or prefix of all the values in a series

### Function: get_validators()

### Function: apply_validators(df, fname, remediation, validators, auto_accept, write_out_file_func)

### Function: lower_case_column(df, column)

### Function: add_suffix(x, suffix)

### Function: remove_common_prefix(x, prefix)

### Function: remove_common_prefix(x, prefix, ws_prefix)

### Function: optional_fn(x)

### Function: add_suffix(x, suffix)

### Function: add_space_start(x)

### Function: lower_case(x)

### Function: format_time(time)

### Function: necessary_fn(x)

### Function: necessary_fn(x)

### Function: optional_fn(x)

### Function: get_long_indexes(d)

### Function: optional_fn(x)

### Function: optional_fn(x)

### Function: lower_case_column_creator(df)

### Function: optional_fn(x)

### Function: optional_fn(x)
