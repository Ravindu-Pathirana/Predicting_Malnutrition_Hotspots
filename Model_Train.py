import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Boosting models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Metrics
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report, confusion_matrix,roc_auc_score

df = pd.read_csv("final_dataset_2000_2022_long_year_order.csv")

# Drop unnecessary columns
df_model = df.drop(columns=["Year"])

# Group by country
df_country = df_model.groupby(['Country', 'Country Code']).mean().reset_index()

print(df_country.shape)
df_country.head()

#Classification labels
def categorize(index):
    if index < 0.15:
        return "Low"
    elif index < 0.30:
        return "Moderate"
    elif index < 0.40:
        return "Hotspot"
    else:
        return "Severe"

df_country["Target"] = df_country["Malnutrition_Index"].apply(categorize)

#Encode target variable
le = LabelEncoder()
df_country["Target_encoded"] = le.fit_transform(df_country["Target"])

# Prepare features and target variable
X = df_country.drop(columns=["Country","Country Code", "Malnutrition_Index", "Target", "Target_encoded","TV_01 - Stunting","TV_02 - Prevalence_of_Overweight","Stunting_norm","Overweight_norm","Raw_norm"])
y = df_country["Target_encoded"]

#Preprocessing pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Model training function
def evaluate(model, X, y):
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train = pd.DataFrame(pipeline.fit_transform(X_train),columns=X.columns)
        X_test = pd.DataFrame(pipeline.transform(X_test),columns=X.columns)
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        
        acc_scores.append(accuracy_score(y_test, pred))
        precision_scores.append(precision_score(y_test, pred, average="weighted"))
        recall_scores.append(recall_score(y_test, pred, average="weighted"))
        roc_auc_scores.append(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        f1_scores.append(f1_score(y_test, pred, average="weighted"))
    
    return np.mean(acc_scores), np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(roc_auc_scores)

#Hyperparameter tuning function
def tune_model(model, param_grid):
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    grid.fit(X_resampled, y_resampled)
    
    print("Best Params:", grid.best_params_)
    
    return grid.best_estimator_

#Random Forest
rf_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

rf_best = tune_model(RandomForestClassifier(random_state=42), rf_params)

#XGBoost
xgb_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

xgb_best = tune_model(XGBClassifier(eval_metric="mlogloss"), xgb_params)

#catboost
cat_params = {
    "model__iterations": [100, 200],
    "model__depth": [4, 6, 8],
    "model__learning_rate": [0.01, 0.1]
}

cat_best = tune_model(CatBoostClassifier(verbose=0), cat_params)

#Decision Tree
dt_params = {
    "model__max_depth": [3, 5, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

dt_best = tune_model(DecisionTreeClassifier(random_state=42), dt_params)

#Gradient Boosting
gb_params = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.1],
    "model__max_depth": [3, 5]
}

gb_best = tune_model(GradientBoostingClassifier(), gb_params)

#KNN
knn_params = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  # Manhattan vs Euclidean
}

knn_best = tune_model(KNeighborsClassifier(), knn_params)

#Logistic Regression
lr_params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"]
}

lr_best = tune_model(LogisticRegression(max_iter=1000), lr_params)

#SVM
svm_params = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"]
}

svm_best = tune_model(SVC(probability=True), svm_params)

#Results compilation
models = {
    "RF": rf_best,
    "XGB": xgb_best,
    "CatBoost": cat_best,
    "DT": dt_best,
    "GB": gb_best,
    "KNN": knn_best,
    "LR": lr_best,
    "SVM": svm_best
}

results = []

for name, model in models.items():
    acc, f1, prec, rec, auc = evaluate(model, X, y)
    
    results.append([name, acc, prec, rec, f1, auc])

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"
])

print(results_df.sort_values(by="F1 Score", ascending=False))
