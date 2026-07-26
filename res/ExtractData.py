# %%
import pandas as pd
import numpy as np
# %%

demo = pd.read_sas("Data\DEMO_J.xpt")
rhq = pd.read_sas("Data\RHQ_J.xpt")
bmx = pd.read_sas("Data\BMX_J.xpt")
smq = pd.read_sas("Data\SMQ_J.XPT")

print("demo_columns:",demo.columns)
print("rhq_columns:",rhq.columns)
print("bmx_columns:",bmx.columns)
print("smq_columns:",smq.columns)
# %%

# %%
# Selected Key Variables (columns) from Each Dataset from each dataset
# %%

# Load your data files
demo = pd.read_sas("Data/DEMO_J.xpt")
rhq = pd.read_sas("Data/RHQ_J.xpt")
bmx = pd.read_sas("Data/BMX_J.xpt")
smq = pd.read_sas("Data/SMQ_J.XPT")

# Print available RHQ columns to confirm
print("Available RHQ columns:")
print([col for col in rhq.columns if col.startswith('RHQ') or col.startswith('RHD')])

# Use only CONFIRMED available columns
demo_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR']  # ID, Age, Gender
rhq_cols = ['SEQN', 'RHQ160']  # Pregnancy history (only confirmed reproductive variable)
bmx_cols = ['SEQN', 'BMXBMI']  # BMI
smq_cols = ['SEQN', 'SMQ020']  # Smoking status

# Merge datasets with error handling
try:
    fertility_df = (
        demo[demo_cols]
        .merge(rhq[rhq_cols], on='SEQN', how='left')
        .merge(bmx[bmx_cols], on='SEQN', how='left')
        .merge(smq[smq_cols], on='SEQN', how='left')
        .query("RIAGENDR == 2")  # Females only
        .rename(columns={
            'RIDAGEYR': 'age',
            'RHQ160': 'ever_pregnant',
            'BMXBMI': 'bmi',
            'SMQ020': 'smoker'
        })
    )
    
    # Convert numeric codes to meaningful values
    fertility_df['ever_pregnant'] = fertility_df['ever_pregnant'].replace({
        1: 'Yes', 2: 'No', 7: np.nan, 9: np.nan  # 7=Refused, 9=Don't know
    })
    
    fertility_df['smoker'] = fertility_df['smoker'].replace({
        1: 'Yes', 2: 'No', 7: np.nan, 9: np.nan
    })
    
    # Create fertility risk categories based on available data
    conditions = [
        fertility_df['age'] >= 45,
        fertility_df['age'] >= 35,
        (fertility_df['bmi'] > 30) | (fertility_df['bmi'] < 18.5),
        fertility_df['smoker'] == 'Yes'
    ]
    
    choices = [
        'Highest Risk (Age 45+)',
        'Moderate Risk (Age 35-44)',
        'Weight-Related Risk',
        'Smoking-Related Risk'
    ]
    
    fertility_df['fertility_risk'] = np.select(conditions, choices, default='Lower Risk')
    
    # Save results
    fertility_df.to_csv('fertility_assessment.csv', index=False)
    print(f"Successfully processed {len(fertility_df)} female participants")
    print("Risk category distribution:")
    print(fertility_df['fertility_risk'].value_counts())

except Exception as e:
    print(f"Error processing data: {str(e)}")
    print("Available columns in each dataset:")
    print("DEMO:", demo.columns.tolist())
    print("RHQ:", rhq.columns.tolist())
# %%
# %%
