
# ============================================
# Import Libraries
# ============================================

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier 

# ============================================
# Load Data
# ============================================

df = pd.read_csv('final_dataset_2000_2022_long_year_order.csv')
print("Data Shape:", df.shape)

# ============================================
# Explore Data
# ============================================
# This to get a small idea about the data in the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

# Identifying columns that has missing values in dataset and missing value length
print(f"Missing value columns in dataset")
print(f"Missing Value Records \n{df.isna().sum()}")

# The missing values already handled using the mean value when creating the dataset, so we don't have to handle missing values here.
# Next let's identify numerical and categorical columns in the dataset
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

print("Numerical Columns")
print(num_cols)
print("Categorical Columns")
print(cat_cols)

# next let's identify unique values in the categorical columns to understand the data better
for col in cat_cols:
    print(f"Unique values in column '{col}': {df[col].unique()}")

# In here only categorical columns present is 'Country' and 'Country Code' . 
# Since both of them means same thing we can drop one of them.
# And we can use label encoding to convert the 'Country' column into numerical values for the ML models.

# ============================================
# Data Preprocessing
# ============================================

def preprocess_data(df):
    df = df.copy()

    for col in ['Country', 'Country Code']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Label encode the 'Country' column
    encoder = LabelEncoder()
    df['Country_encoded'] = encoder.fit_transform(df['Country'])

    # since Malnutrition_Index is in as numerical value but we want to predict whether it is LOW, MODERATE, HOTSPOT, SEVERE HOTSPOT, we can create a new column 'Malnutrition_Level' based on the value of 'Malnutrition_Index'
    def malnutrition_level(index):
        if index < 0.15:
            return 0        # LOW Malnutrition
        elif index < 0.30:
            return 1        # MODERATE Malnutrition
        elif index < 0.40:
            return 2        # HOTSPOT Malnutrition
        else:
            return 3        # SEVERE HOTSPOT Malnutrition

  
    df['Malnutrition_Level'] = df['Malnutrition_Index'].apply(malnutrition_level)


    return df

# Preprocess the data
df = preprocess_data(df)

# ============================================
#  Feature Engineering
# ============================================
def feature_engineering(df):
    df = df.copy()

    # first let's drop some unneccary values in the datastet that we won't be using for the ML models
    unnecessary_columns = ['Country', 'Country Code', 'Malnutrition_Index', 'TV_01 - Stunting',
       'TV_02 - Prevalence_of_Overweight', 'Stunting_norm', 'Overweight_norm',
       'Raw_norm']
    for col in unnecessary_columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    
    df['Economic_Score'] = np.log1p(df['Predictor_04 - GDP per capita(current US$)']) * df['Predictor_05 - HDI Value'] # This helps to increase the accuracy of the model.

    return df

# Perform feature engineering
df = feature_engineering(df)
print(df.head())


# ============================================
#  Define Features & Target
# ============================================
TARGET = 'Malnutrition_Level'  
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Fill any remaining missing values in features to keep sklearn models stable.
X = X.fillna(X.median(numeric_only=True))
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].mode(dropna=True).iloc[0])

# ============================================
# Train Validation Split
# ============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# Model Training
# ============================================

"""Here we will train multiple ML models and evaluate their performance on the validation set. 
  We will use accuracy, precision, recall, F1-score, and ROC-AUC as evaluation metrics to compare the models. 
  We will also perform hyperparameter tuning for the best performing model to further improve its performance."""
models = {
    'Logistic Regression': LogisticRegression(
        solver='newton-cg',
        max_iter=1000,
        class_weight='balanced',
        tol = 0.01,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=1600,
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=1500,
        random_state=42,
    ),
    'SVM': SVC(
        probability=True, 
        random_state=42,
        class_weight='balanced'
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42,
    ),
    'KNN': KNeighborsClassifier(),
    'LightGBM': LGBMClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    'CatBoost': CatBoostClassifier( 
        verbose=0,
        random_state=42,
        auto_class_weights='Balanced',
        allow_writing_files=False
    )
}

best_model = None
best_score = 0

results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr') if y_proba is not None else 'N/A'

    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc}")

    results.append({
        'Name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    })

    if accuracy > best_score:
        best_score = accuracy
        best_model = model

print('Best Model:', best_model)
print('Best Accuracy:', best_score)

result_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
print(result_df)
result_df.to_csv('model_comparison_results.csv', index=False)