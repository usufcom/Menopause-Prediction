# %%
"""
NHANES Fertility Assessment Tool
--------------------------------
This script analyzes NHANES 2017-2018 data to assess female fertility risk factors,
including age, BMI, smoking status, and pregnancy history, with estimated menstrual status.
"""

import pandas as pd
import numpy as np

# DATA LOADING
# %%
def load_nhanes_data():
    """Load NHANES 2017-2018 data files from local directory"""
    try:
        demo = pd.read_sas("Data/DEMO_J.xpt")
        rhq = pd.read_sas("Data/RHQ_J.xpt")
        bmx = pd.read_sas("Data/BMX_J.xpt")
        smq = pd.read_sas("Data/SMQ_J.XPT")
        
        print("Data loaded successfully")
        return demo, rhq, bmx, smq
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

# DATA PROCESSING
def process_fertility_data(demo, rhq, bmx, smq):
    """
    Process and merge NHANES data into fertility assessment dataframe
    Args:
        demo, rhq, bmx, smq: Raw NHANES dataframes
    Returns:
        Processed DataFrame with fertility metrics
    """
    # Confirm required columns exist
    required_cols = {
        'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
        'RHQ': ['SEQN', 'RHQ160'],
        'BMX': ['SEQN', 'BMXBMI'],
        'SMQ': ['SEQN', 'SMQ020']
    }
    
    # Verify columns
    for df, cols in zip([demo, rhq, bmx, smq], required_cols.values()):
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

    # Merge datasets
    fertility_df = (
        demo[required_cols['DEMO']]
        .merge(rhq[required_cols['RHQ']], on='SEQN', how='left')
        .merge(bmx[required_cols['BMX']], on='SEQN', how='left')
        .merge(smq[required_cols['SMQ']], on='SEQN', how='left')
        .query("RIAGENDR == 2")  # Filter females only
        .rename(columns={
            'RIDAGEYR': 'age',
            'RHQ160': 'ever_pregnant',
            'BMXBMI': 'bmi',
            'SMQ020': 'smoker'
        })
        .dropna(subset=['age'])  # Require age data
    )
    
    # Clean categorical variables
    fertility_df['ever_pregnant'] = fertility_df['ever_pregnant'].replace({
        1: 'Yes', 2: 'No', 7: np.nan, 9: np.nan  # 7=Refused, 9=Don't know
    })
    
    fertility_df['smoker'] = fertility_df['smoker'].replace({
        1: 'Yes', 2: 'No', 7: np.nan, 9: np.nan
    })
    
    return fertility_df

# MENSTRUAL STATUS ESTIMATION
# %%
def estimate_menstrual_status(age):
    """
    Estimate menstrual status based on age using clinical guidelines
    Args:
        age: Participant age in years
    Returns:
        Tuple of (status, probability, confidence)
    """
    if pd.isna(age):
        return np.nan, np.nan, np.nan
    
    # Clinical thresholds based on STRAW+10 staging
    if age >= 55:
        return 'Postmenopausal', 0.95, 'High'
    elif 45 <= age < 55:
        prob = min(0.7 + (age-45)*0.05, 0.9)  # 70% at 45, 90% at 49
        return 'Perimenopausal', prob, 'Moderate'
    elif 40 <= age < 45:
        return 'Premenopausal (possible early transition)', 0.3, 'Low'
    else:
        return 'Premenopausal', 0.01, 'High'

# RISK ASSESSMENT
# %%
def calculate_fertility_risk(df):
    """
    Calculate comprehensive fertility risk score
    Args:
        df: Processed fertility dataframe
    Returns:
        DataFrame with added risk assessments
    """
    # Add menstrual status estimates
    status_data = df['age'].apply(lambda x: pd.Series(estimate_menstrual_status(x)))
    df[['menstrual_status', 'menopause_prob', 'status_confidence']] = status_data
    
    # Calculate risk factors
    conditions = [
        df['age'] >= 45,
        df['age'] >= 35,
        (df['bmi'] > 30) | (df['bmi'] < 18.5),
        df['smoker'] == 'Yes',
        df['ever_pregnant'] == 'Yes'
    ]
    
    choices = [
        'Highest Risk (Age 45+)',
        'Moderate Risk (Age 35-44)',
        'Weight-Related Risk',
        'Smoking-Related Risk',
        'Parity-Related Factor'
    ]
    
    df['fertility_risk'] = np.select(conditions[:4], choices[:4], default='Lower Risk')
    df['risk_factors'] = df.apply(lambda x: ', '.join([choices[i] for i, cond in enumerate(conditions) if cond[x.name]]), axis=1)
    
    # Composite risk score (0-100)
    df['risk_score'] = (
        np.where(df['age'] >= 35, (df['age'] - 35) * 2, 0) +  # Age contributes 0-30 points
        np.where(df['bmi'] < 18.5, 10, 0) +
        np.where(df['bmi'] > 30, 15, 0) +
        np.where(df['smoker'] == 'Yes', 20, 0) +
        np.where(df['age'] >= 45, 30, 0)
    ).clip(0, 100)
    
    return df

# MAIN EXECUTION
# %%
if __name__ == "__main__":
    print("NHANES Fertility Assessment Pipeline")
    print("-----------------------------------")
    
    # Load data
    demo, rhq, bmx, smq = load_nhanes_data()
    if demo is None:
        exit()
    
    try:
        # Process data
        fertility_df = process_fertility_data(demo, rhq, bmx, smq)
        
        # Add assessments
        fertility_df = calculate_fertility_risk(fertility_df)
        
        # Save results
        output_file = 'fertility_assessment_comprehensive.csv'
        fertility_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nAnalysis completed successfully. Results saved to {output_file}")
        print(f"Participants analyzed: {len(fertility_df)}")
        print("\nRisk Category Distribution:")
        print(fertility_df['fertility_risk'].value_counts())
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Available RHQ columns:", [c for c in rhq.columns if c.startswith('RHQ')])
# %%
fertility_df
# %%
