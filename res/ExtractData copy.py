# %%
import pandas as pd
import numpy as np
import requests
from io import BytesIO
# %%

def download_nhanes_file(url, component_name):
    """Download a single NHANES data file"""
    try:
        print(f"Downloading {component_name} data from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_sas(BytesIO(response.content), format='xport')
        print(f"Successfully downloaded {component_name} data")
        return df
        
    except Exception as e:
        print(f"Failed to download {component_name} data: {str(e)}")
        return None

def clean_demo_data(df):
    """Clean demographic data"""
    demo_clean = df[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDHHIN2']].copy()
    demo_clean.columns = ['id', 'age', 'gender', 'race', 'education', 'income']
    return demo_clean.query("gender == 2")  # Keep only female participants

def calculate_fertility_status(df):
    """Calculate fertility status based on available variables"""
    conditions = [
        df['menopausal'] == 1,  # Confirmed menopause
        df['age'] >= 50,
        (df['age'] >= 45) & (df['cycle_regularity'] == 3),
        (df['age'] >= 38) & (df['cycle_regularity'] == 1),
        df['age'] >= 35,
        df['cycle_regularity'] == 1,
        (df['bmi'] < 18.5) | (df['bmi'] > 30),
        df['smoking_status'] == 1
    ]
    
    choices = [
        'Post-Menopausal',
        'Very Low (Probable Menopause)',
        'Low (Perimenopausal)',
        'Moderate-Low (Aging/Ovulatory Dysfunction)',
        'Age-Related Decline',
        'Moderate (Cycle Irregularities)',
        'Moderate Risk (Weight)',
        'Moderate Risk (Smoking)'
    ]
    
    df['fertility_status'] = np.select(conditions, choices, default='Normal')
    return df

def main():
    # NHANES 2017-2018 Data URLs
    data_urls = {
        'DEMO': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt',
        'RHQ': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/RHQ_J.xpt',
        'BMX': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt',
        'SMQ': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.XPT',
        'EST': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_EST_J.XPT'
    }

    # Download all files
    print("Starting data download...")
    nhanes_data = {}
    for component, url in data_urls.items():
        df = download_nhanes_file(url, component)
        if df is not None:
            nhanes_data[component] = df

    if not nhanes_data:
        print("No data downloaded. Check internet connection and URLs.")
        return

    print("\nProcessing data...")
    try:
        # Clean and merge core data
        merged = (
            clean_demo_data(nhanes_data['DEMO'])
            .merge(nhanes_data['RHQ'][['SEQN', 'RHQ030', 'RHQ042', 'RHQ050']], on='SEQN')
            .merge(nhanes_data['BMX'][['SEQN', 'BMXBMI']], on='SEQN')
            .merge(nhanes_data['SMQ'][['SEQN', 'SMQ020']], on='SEQN')
        )
        
        # Rename columns
        fertility_df = merged.rename(columns={
            'RHQ030': 'menarche_age',
            'RHQ042': 'menopausal',
            'RHQ050': 'cycle_regularity',
            'BMXBMI': 'bmi',
            'SMQ020': 'smoking_status'
        })
        
        # Add estradiol if available
        if 'EST' in nhanes_data:
            fertility_df = fertility_df.merge(
                nhanes_data['EST'][['SEQN', 'LBXEST']].rename(columns={'LBXEST': 'estradiol'}),
                on='SEQN', how='left'
            )
        
        # Calculate fertility status
        fertility_df = calculate_fertility_status(fertility_df)
        
        # Save results
        output_file = 'nhanes_fertility_predictions.csv'
        fertility_df.to_csv(output_file, index=False)
        print(f"\nSuccess! Saved {len(fertility_df)} records to {output_file}")
        print("Columns included:", fertility_df.columns.tolist())
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Available data components:", list(nhanes_data.keys()))

# %%
if __name__ == "__main__":
    main()
# %%