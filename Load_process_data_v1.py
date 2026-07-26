# %%
import pyreadstat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os
import re

# %%
def get_process_Data(data_dir = "Data", filename = "RHQ_J.xpt"):
    # Load the Reproductive Health datasets
    # https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2017
    #data_dir = "Data"
    RH_file_path = os.path.join(data_dir, filename)

    Reproductive_Health_df, meta = pyreadstat.read_xport(RH_file_path)
    labels_RH = meta.column_names_to_labels


    # Load the Demographics datasets
    Demo_file_path = os.path.join(data_dir, "DEMO_J.xpt")
    Demographic_df, meta = pyreadstat.read_xport(Demo_file_path)
    labels_DM = meta.column_names_to_labels

    # Preview data
    #print(Reproductive_Health_df.head())
    # Print variable name and description
    #for name, label in labels_RH.items():
    #   print(f"{name}: {label}")

    # %%

    # Merge the age column from Demographic_df into Reproductive_Health_df
    Reproductive_Health_df = pd.merge(
        Reproductive_Health_df,
        Demographic_df[['SEQN', 'RIDAGEYR']],  # Just get the columns you need
        on='SEQN',
        how='left'  # keep all rows in Reproductive_Health_df
    )

    # Add label for RIDAGEYR
    labels_RH['RIDAGEYR'] = "Age in years at screening"

    # Preview data
    #display(Reproductive_Health_df.tail())
    #Reproductive_Health_df.shape

    # Isolate only those who reported menopause as the reason for no periods
    #Reproductive_Health_df = Reproductive_Health_df[Reproductive_Health_df['RHD043'] == 7].copy()
    #Reproductive_Health_df = Reproductive_Health_df.drop(columns=['RHD043'])
    print('Reproductive_Health_df shape:',Reproductive_Health_df.shape)

    # %%
    ### Drop redundant and not useful columns
    # List of attributes to exclude
    columns_to_exclude = ['RHD280', 'RHQ291', 'RHQ305', 'RHQ332']

    category_lookup = {
        'RHD280_Had_hysterectomy': {
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'RHQ305_Had_both_ovaries_removed': {
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        }
    }

    # Keep only participants with NO hysterectomy AND NO ovary removal
    Reproductive_Health_df = Reproductive_Health_df[
        (Reproductive_Health_df['RHD280'] == 2) & (Reproductive_Health_df['RHQ305'] == 2)
    ]

    # Then drop the redundant columns, but only if they exist in the dataframe
    columns_to_exclude = ['RHD280', 'RHQ291', 'RHQ305', 'RHQ332']
    Reproductive_Health_df = Reproductive_Health_df.drop(
        columns=[col for col in columns_to_exclude if col in Reproductive_Health_df.columns]
    )
    # %%

    ### Load other datasets

    # 1) Standardized metadata dict
    nhanes_datasets = {
        "ALQ_J": {
            "file": "ALQ_J.xpt",
            "columns2extract": [
                "SEQN", "ALQ121", "ALQ130", "ALQ142",
                "ALQ270", "ALQ280", "ALQ290", "ALQ151"
            ]
        },
        "DIQ_J": {
            "file": "DIQ_J.xpt",
            "columns2extract": ["SEQN", "DIQ010", "DID040", "DBQ700"]
        },
        "DBQ_J": {
            "file": "DBQ_J.xpt",
            "columns2extract": ["SEQN", "DBD895", "DBD900", "DBD905", "DBD910"]
        },
        "INQ_J": {
            "file": "INQ_J.xpt",
            "columns2extract": ["SEQN", "IND235", "IND310"]
        },
        "MCQ_J": {
            "file": "MCQ_J.xpt",
            "columns2extract": [
                "SEQN", "MCQ160M", "MCQ170M", "MCD180M", "MCQ220",
                "MCQ230A", "MCD240A", "MCQ230B", "MCD240B",
                "MCQ230C", "MCD240C", "MCQ230D"
            ]
        },
        "PUQMEC_J": {
            "file": "PUQMEC_J.xpt",
            "columns2extract": ["SEQN", "PUQ100", "PUQ110"]
        },
        "SMQ_J": {
            "file": "SMQ_J.XPT",
            "columns2extract": ["SEQN", "SMQ020", "SMD030", "SMQ621"]
        },
        "WHQ_J": {
            "file": "WHQ_J.xpt",
            "columns2extract": ["SEQN", "WHD010", "WHD020", "WHD120", "WHD130"]
        }
    }


    extra_columns = []

    for dataset in nhanes_datasets.values():
        columns = dataset.get("columns2extract", [])
        filtered_columns = [col for col in columns if col != "SEQN"]
        extra_columns.extend(filtered_columns)

    # print(extra_columns)
    # %%
    # 2) Merge loop
    merged_df     = Reproductive_Health_df.copy()
    merged_labels = labels_RH.copy()

    for key, info in nhanes_datasets.items():
        path = os.path.join(data_dir, info["file"])
        if not os.path.exists(path):
            print(f"⚠️ Missing file: {path}")
            continue

        extra_df, meta = pyreadstat.read_xport(path)

        # restrict to just the columns you listed
        cols = info["columns2extract"]
        # make sure SEQN is first (and exists)
        cols = ["SEQN"] + [c for c in cols if c != "SEQN" and c in extra_df.columns]
        extra_df = extra_df[cols]

        # drop any accidental duplicate SEQN
        extra_df = extra_df.drop_duplicates(subset="SEQN")

        # merge one-to-one on SEQN
        before = merged_df.shape
        merged_df = pd.merge(merged_df, extra_df, on="SEQN", how="left")

        # collect labels
        merged_labels.update(meta.column_names_to_labels)
        
        #print(f"{key}: +{extra_df.shape[1]-1} cols → merged_df is now {merged_df.shape} (was {before})")

    # final shapes
    print("Final merged_df shape:", merged_df.shape)

    # %%
    # Create the summary list
    summary = []

    for col in merged_df.columns:
        desc = merged_labels.get(col, "No description available")  # Safely get description
        count = merged_df[col].count()         # Non-null count
        summary.append({"Column": col, "Description": desc, "Non-null Count": count})

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)

    # Optional: Show all rows when printing
    pd.set_option("display.max_rows", None) 


    # Preview first few rows
    #print(summary_df)

    # Set stricter MVP threshold (to remove columns with NaN values greater than the threshold)
    threshold = int(0.40*merged_df.shape[0])

    # Condensed and more populated menopause-relevant columns
    mvp_menopause_important = [
        'RHQ031',   # Had regular periods in past 12 months 
        'RHQ060',   # Age at last menstrual period 
        'RHQ540',   # Ever use female hormones 
        'RIDAGEYR', # Age in years at screening 
    ]

    # Filter based on threshold or core clinical value
    columns_to_keep = summary_df[
        (summary_df["Non-null Count"] >= threshold) |
        (summary_df["Column"].isin(mvp_menopause_important))
    ]["Column"].tolist()

    # Filter dataset
    filtered_RH_df = merged_df[columns_to_keep]

    print(f"Retained {len(filtered_RH_df.columns)} columns out of {merged_df.shape[1]}")

    #nan_counts1 = filtered_RH_df.isna().sum()
    #print("\nNaN count in remaining columns:",nan_counts1)

    # %%
    ### Drop redundant and not useful columns
    # List of attributes to exclude
    columns_to_exclude = ['RHD280', 'RHQ291', 'RHQ305', 'RHQ332']

    # Drop the redundant columns, but only if they exist in the dataframe
    filtered_RH_df = filtered_RH_df.drop(
        columns=[col for col in columns_to_exclude if col in filtered_RH_df.columns]
    )

    # -----------------------------
    # Filter summary table
    # -----------------------------
    # Remove the excluded columns from the columns_to_keep list
    columns_to_keep = [col for col in columns_to_keep if col not in columns_to_exclude]

    filtered_summary_df = summary_df[summary_df["Column"].isin(columns_to_keep)]

    #print('length of filtered_summary_df:',len(columns_to_keep))
    #print(filtered_summary_df.sort_values(by="Non-null Count", ascending=False).reset_index(drop=True))
    # %%
    # Define patterns that suggest numerical columns
    numerical_keywords = [
        'age', 'how many', 'number of', 'times', 'weigh',
        'live birth', '# of', 'income', 'height', '# days',
        'how often', '# alcohol', 'savings'
    ]

    # Initialize lists
    numerical_cols = []
    categorical_cols = []
    column_descriptions = {}

    # Loop through filtered_summary_df
    for _, row in filtered_summary_df.iterrows():
        col = row['Column']
        desc = row['Description'].strip()
        
        if col == 'SEQN':  # Skip identifier column
            continue
        
        # Check if description suggests numerical
        desc_lower = desc.lower()
        is_numerical = any(kw in desc_lower for kw in numerical_keywords)

        # Append to appropriate list
        if is_numerical:
            tag = 'num.'
            numerical_cols.append(col)
        else:
            tag = 'cat.'
            categorical_cols.append(col)
        
        # Add to description mapping
        column_descriptions[col] = f"{col}: {desc}"
        #print(f"{tag} : {column_descriptions[col]}\n")
    # %%
    df_classification = filtered_RH_df.copy()

    # First Replace wrong values (999.0, 777.0, 77.0) with NaN in the numerical columns
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(9999.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(7777.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(6666.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(5555.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(999.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(777.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(99.0, np.nan)
    df_classification[numerical_cols] = df_classification[numerical_cols].replace(77.0, np.nan)
    #df_classification['RHQ172'] = df_classification['RHQ172'].replace(9.0, np.nan)
    # %%

    numerical_cols_age_related = ['RHQ160', 'RHD180', 'RHD190'] 

    for col in numerical_cols_age_related:
        # Group by age and fill NaNs with median of the group
        df_classification[col] = df_classification.groupby('RIDAGEYR')[col]\
            .transform(lambda x: x.fillna(x.median()))
        
        
    # See how many NaNs are left in the numerical columns
    nan_counts = df_classification[numerical_cols].isna().sum()
    #print("\nNaN count in numerical_cols columns:")
    #print(nan_counts[nan_counts > 0].sort_values(ascending=False))

    # Replace the remaining NaNs in numerical data With the global median of each column
    # Loop over all columns
    for col in df_classification.columns:
        # Only apply to numerical columns NOT in category_lookup
        if col in numerical_cols:
            if df_classification[col].dtype in ['float64', 'int64']:
                median_value = df_classification[col].median()
                df_classification[col] = df_classification[col].fillna(median_value)

    # See how many NaNs are left in the numerical columns
    nan_counts = df_classification[numerical_cols].isna().sum()
    #print("\nNaN count in numerical_cols columns:")
    #print(nan_counts[nan_counts > 0].sort_values(ascending=False))


    # Set the labels for each column based on 
    # the description from https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/RHQ_J.htm 
    category_lookup = {
        'RHQ131_Ever_Been_Pregnant': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'RHQ162_Diabetes_During_Pregnancy': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'RHQ420_Taken_Birth_Control_Pills': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'RHQ540_Used_Female_Hormones': {
            0: 'Other ',
            1: 'Yes',
            2: 'No'
        },
        'ALQ151_Ever_have_4_or_more_drinks_every_day': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'DIQ010_Doctor_told_you_have_diabetes': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            3: 'Borderline',
            7: 'Refused',
            9: 'Do not know'
        },
        'MCQ160M_Ever_told_you_had_thyroid_problem': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'MCQ220_Ever_told_you_had_cancer_or_malignancy': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'PUQ100_Used_chemical_products_in_home_to_control_insects': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'PUQ110_Used_chemical_products_to_kill_weeds': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        },
        'SMQ020_Smoked_at_least_100_cigarettes_in_life': {
            0: 'Other ',
            1: 'Yes',
            2: 'No',
            7: 'Refused',
            9: 'Do not know'
        }
    }


    # Mapping from short codes to descriptive column names
    column_rename_map = {
        'RHQ131': 'RHQ131_Ever_Been_Pregnant',
        'RHQ162': 'RHQ162_Diabetes_During_Pregnancy',
        'RHQ420': 'RHQ420_Taken_Birth_Control_Pills',
        'RHQ540': 'RHQ540_Used_Female_Hormones',
        'ALQ151': 'ALQ151_Ever_have_4_or_more_drinks_every_day',
        'DIQ010': 'DIQ010_Doctor_told_you_have_diabetes',
        'MCQ160M': 'MCQ160M_Ever_told_you_had_thyroid_problem',
        'MCQ220': 'MCQ220_Ever_told_you_had_cancer_or_malignancy',
        'PUQ100': 'PUQ100_Used_chemical_products_in_home_to_control_insects',
        'PUQ110': 'PUQ110_Used_chemical_products_to_kill_weeds',
        'SMQ020': 'SMQ020_Smoked_at_least_100_cigarettes_in_life'
    }


    # Copy and rename the columns in the DataFrame
    df_classification_copy = df_classification.copy()
    df_classification_copy.rename(columns=column_rename_map, inplace=True)

    # Count occurrences of each label in the categorical columns using the updated lookup table
    for col, mapping in category_lookup.items():
        if col in df_classification_copy.columns:
            #print(f"\n{col}: Label Counts")
            label_series = df_classification_copy[col].map(mapping)
            #print(label_series.value_counts(dropna=False))

    # %%
    #df_classification1 = df_classification.dropna()
    # Drop columns with more than 60% missing data
    # Columns before dropping
    original_cols = set(df_classification.columns)

    threshold = 0.7
    df_classification = df_classification.dropna(axis=1, thresh=int((1 - threshold) * len(df_classification)))
    #print(df_classification.shape)

    # Columns after dropping
    remaining_cols = set(df_classification.columns)

    # Dropped columns
    dropped_columns = original_cols - remaining_cols
    #print(f"Dropped Columns (more than {threshold*100}% missing):")
    #print(dropped_columns)


    nan_counts = df_classification.isna().sum()
    #print("\nNaN count in remaining columns:")
    #print(nan_counts[nan_counts > 0].sort_values(ascending=False))

    # %%
    # Values to treat as "missing"
    missing_codes = [7, 9]

    for col in categorical_cols:
        if col in df_classification.columns:
            # Replace specific codes with 0
            df_classification[col] = df_classification[col].replace(missing_codes, 0)
            # Replace NaNs with 0
            df_classification[col] = df_classification[col].fillna(0)
            #print('replaced!!!!')

    nan_counts = df_classification.isna().sum()
    #print("\nNaN count in remaining columns:")
    #print(nan_counts[nan_counts > 0].sort_values(ascending=False))

    df_classification = df_classification.fillna(0)

    print('df_classification.shape:',df_classification.shape)

    # Done! NaNs, Refused, and Don't Know coded as 0


    # %%
    return summary_df, filtered_summary_df, df_classification 
