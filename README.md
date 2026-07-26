# Menopause Prediction

Machine learning prototype to estimate menopause status and risk using NHANES 2017–2018 reproductive health, demographic, and lifestyle data.

Clinically, menopause is defined as 12 consecutive months without a menstrual period (excluding pregnancy, illness, or medication). This project builds an MVP that models that transition probabilistically from survey and health questionnaire responses.

## Overview

The workflow has two complementary stages:

1. **Population-level growth curve** — logistic regression models how menopause probability rises with age (and risk factors like smoking), producing interpretable S-shaped risk curves.
2. **Individual-level classification** — supervised ML models predict menopause status from a richer feature set, enabling personalized risk scores and threshold-based age estimates.

Final cohort after filtering: **1,901 participants**, **40 features**.

## Data & preprocessing

Raw data comes from the [NHANES 2017–2018 cycle](https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/Default.aspx?BeginYear=2017) (CDC), loaded from SAS Transport (`.xpt`) files via **pyreadstat**.

| Dataset | Content |
|---|---|
| `RHQ_J` | Reproductive health questionnaire (primary) |
| `DEMO_J` | Demographics (age) |
| `ALQ_J`, `DIQ_J`, `DBQ_J`, `INQ_J`, `MCQ_J`, `PUQMEC_J`, `SMQ_J`, `WHQ_J` | Alcohol, diabetes, diet, income, medical conditions, pesticides, smoking, weight history |

Preprocessing is handled in `Load_process_data_v1.py`. See [Exploratory data analysis & data cleaning](#exploratory-data-analysis--data-cleaning) for the full EDA and cleaning workflow.

## Exploratory data analysis & data cleaning

All EDA and cleaning logic lives in `Load_process_data_v1.py`. The function returns three artifacts used throughout the notebook:

| Output | Purpose |
|---|---|
| `summary_df` | Every merged column with NHANES description and non-null count |
| `filtered_summary_df` | Retained columns after feature selection, tagged numerical vs categorical |
| `df_classification` | Cleaned, imputed feature matrix ready for modeling |

### Cohort shaping

Starting from the full reproductive-health sample, rows are filtered to women with **no hysterectomy** (`RHD280 == 2`) and **no bilateral oophorectomy** (`RHQ305 == 2`), so the outcome reflects natural menopause rather than surgical causes.

| Step | Shape |
|---|---|
| Reproductive health + age, after surgical-history filter | 3,286 × 48 |
| After merging 8 additional NHANES modules | 1,901 × 79 |
| After column selection & cleaning | **1,901 × 40** |

### EDA steps

1. **Variable inventory** — NHANES variable labels from `pyreadstat` metadata are joined to each column to build `summary_df` (column name, plain-language description, non-null count).

2. **Missingness audit** — Columns are ranked by non-null count. Any variable below **60% populated** is flagged for removal unless it is clinically essential for menopause modeling.

3. **Feature retention rules** — A column is kept if either:
   - Non-null count ≥ 40% of rows, **or**
   - It is in the MVP menopause set: `RHQ031`, `RHQ060`, `RHQ540`, `RIDAGEYR`

   This reduced the feature set from **79 → 40 columns**.

4. **Variable type classification** — Using NHANES descriptions, columns are tagged as **numerical** (keywords: age, weight, height, income, number of, etc.) or **categorical** (yes/no/refused survey items). Results are stored in `filtered_summary_df`.

5. **Categorical value exploration** — NHANES coded responses (1 = Yes, 2 = No, 7 = Refused, 9 = Don't know) are mapped to readable labels and inspected via value counts before encoding.

6. **Target construction** — In the notebook, binary `Menopausal_Status` is derived from `RHQ031`: `1` when the participant reports no regular periods in the past 12 months (`RHQ031 == 2`).

### Data cleaning steps

1. **Duplicate removal** — Duplicate `SEQN` values are dropped when merging auxiliary datasets to enforce one row per participant.

2. **Clinical exclusion columns dropped** — `RHD280`, `RHQ291`, `RHQ305`, `RHQ332` are removed after filtering (they were only needed for the exclusion step).

3. **NHANES special codes → missing (numerical)** — Standard NHANES sentinel values are recoded to `NaN`:

   | Code | Meaning |
   |---|---|
   | 77, 777, 7777 | Refused |
   | 99, 999, 9999 | Don't know |
   | 5555, 6666 | Other NHANES missing codes |

4. **Imputation (numerical)** — Two-step strategy:
   - Age-related fields (`RHQ160`, `RHD180`, `RHD190`): median imputation **within age group** (`RIDAGEYR`)
   - All remaining numerical gaps: **global column median**

5. **High-missingness column drop** — Columns with **>70% missing** after imputation are removed entirely.

6. **Categorical encoding** — Refused (`7`) and Don't know (`9`) responses are recoded to `0`; remaining NaNs in categorical columns are filled with `0`. A final pass fills any residual NaNs with `0` so the matrix is fully numeric for ML pipelines.

7. **Preprocessed export** — The cleaned dataset is saved as `df_classification.csv` for fast reload without re-running the full pipeline.

### What the cleaned data looks like

- **1,901 participants**, **40 features**, no missing values
- Mix of reproductive history, lifestyle (smoking, alcohol), medical history (diabetes, thyroid, cancer), weight/height at age 25, and income
- Ready for both statsmodels logistic regression and scikit-learn classification pipelines
## Modeling approach

### Stage 1 — Growth curve (statsmodels)

- **Univariate logistic regression**: `Menopausal_Status ~ RIDAGEYR`
- **Multivariate extension**: adds smoking (`SMQ020`) and weight at age 25 (`WHD020`)
- Age bins (2-year intervals) compared against predicted probabilities for calibration plots
- Pseudo R² ≈ 0.63 for the age-only model — age is the dominant predictor

### Stage 2 — Individual classification (scikit-learn)

**Features used for ML models:**

| Variable | Description |
|---|---|
| `RIDAGEYR` | Age at screening |
| `RHQ010` | Age at first menstrual period |
| `RHQ420` | Ever taken birth control pills |
| `WHD120` | Self-reported weight at age 25 (lbs) |
| `WHD130` | Self-reported height at age 25 (in) |
| `SMQ020` | Smoked ≥100 cigarettes in lifetime |

**Pipeline:**

```
ColumnTransformer
├── StandardScaler  → numerical features
└── OneHotEncoder   → categorical features
        ↓
Classifier (Logistic Regression | Random Forest | SVM | Gradient Boosting | MLP)
```

- **Train/test split**: 80/20, stratified on target
- **Hyperparameter tuning**: `GridSearchCV` with 5-fold stratified cross-validation
- **Evaluation metrics**: accuracy, macro F1, ROC AUC
- **Best model (MLP)** used for patient-level risk prediction and age-at-threshold estimation

## Techniques used

| Area | Tools / methods |
|---|---|
| Data I/O | pyreadstat, pandas |
| EDA & visualization | matplotlib, seaborn |
| Statistical modeling | statsmodels (logistic regression) |
| Machine learning | scikit-learn (pipelines, GridSearchCV, StratifiedKFold) |
| Classifiers | Logistic Regression, Random Forest, SVM (RBF/linear), HistGradientBoosting, MLP |
| Preprocessing | StandardScaler, OneHotEncoder, ColumnTransformer |
| Risk interpretation | Predicted probability curves, 50%/90% threshold crossing |

## Results

### Binned Observations vs. Logistic Fit

![Binned Observations vs. Logistic Fit](images/Binned%20Observations%20vs.%20Logistic%20Fit.png)

Logistic regression fitted on age closely tracks the observed data. The S-shaped curve shows menopause probability rising sharply between the mid-40s and mid-50s, with 50% probability around age 50 — consistent with the typical age of menopause in the study population.

### Classification Results

![Classification Results](images/Classification%20Results.png)

Five classifiers were compared on a held-out test set. All models performed strongly (accuracy > 91%, ROC AUC > 0.94). **MLP** achieved the highest accuracy (93.2%) and F1 score (93.0%), while **Logistic Regression** achieved the highest ROC AUC (0.960). These results indicate that age and clinical covariates are highly predictive of menopause status.

| Model | Accuracy | F1 Score (macro) | ROC AUC |
|---|---|---|---|
| MLP | 0.932 | 0.930 | 0.956 |
| Logistic Regression | 0.929 | 0.928 | 0.960 |
| Random Forest | 0.929 | 0.928 | 0.945 |
| SVM | 0.921 | 0.919 | 0.955 |
| Gradient Boosting | 0.911 | 0.909 | 0.951 |

### Predicted Menopause Risk Curve

![Predicted Menopause Risk Curve](images/Predicted%20Menopause%20Risk%20Curve.png)

Stratifying the logistic model by smoking status shows that smokers reach the same menopause probability at a younger age than non-smokers. The smoker curve is shifted roughly 3 years earlier, with 50% probability around age 51 for smokers vs. age 54 for non-smokers — aligning with epidemiological evidence that smoking accelerates ovarian aging.

### Estimated Menopause Risk Curve

![Estimated Menopause Risk Curve](images/Estimated%20Menopause%20Risk%20Curve.png)

Patient-level risk curve derived from the trained MLP model, with threshold markers at 50% and 90% probability. For this profile, the model estimates **50% menopause risk at age 50.1** and **90% risk at age 55.2**, highlighting the steep transition window between ages 45 and 55.

## Project structure

```
Menopause_Prediction/
├── Load_process_data_v1.py                    # NHANES data loading and preprocessing
├── Menopause_analysis_and Prediction_V6.ipynb # Analysis, modeling, and visualization
├── df_classification.csv                      # Preprocessed feature matrix
├── Data/                                      # NHANES .xpt source files
├── images/                                    # Result figures
└── README.md
```

## How to run

1. Place NHANES `.xpt` files in the `Data/` folder (see [NHANES data page](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2017))
2. Install dependencies: `pip install pyreadstat pandas numpy matplotlib seaborn statsmodels scikit-learn`
3. Open and run `Menopause_analysis_and Prediction_V6.ipynb`, or preprocess standalone:

```python
from Load_process_data_v1 import get_process_Data
summary_df, filtered_summary_df, df_classification = get_process_Data(
    data_dir="Data", filename="RHQ_J.xpt"
)
```

Alternatively, load the preprocessed CSV directly: `df_classification.csv`.
