import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load data
df = pd.read_csv("loan_default.csv")

# Drop ID
df = df.drop(columns=["LoanID"], errors="ignore")

# Target
target = "Default"

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# Features & target
X = df.drop(columns=[target])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

#  5 Boosting Models
model1 = GradientBoostingClassifier()
model2 = AdaBoostClassifier(n_estimators=200)
model3 = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
model4 = LGBMClassifier(n_estimators=200, verbose=-1)
model5 = CatBoostClassifier(verbose=0)

# Voting Ensemble
voting_model = VotingClassifier(
    estimators=[
        ("gb", model1),
        ("ada", model2),
        ("xgb", model3),
        ("lgbm", model4),
        ("cat", model5)
    ],
    voting="soft"   # use probabilities
)

# Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", voting_model)
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print("\nMODEL PERFORMANCE")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve (save)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")

# Save model
joblib.dump(pipeline, "loan_pipeline.pkl")
print("\nModel saved successfully!")
