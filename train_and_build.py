import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================= LOAD DATA =================
df = pd.read_csv("smartphones.csv")
print("Dataset loaded:", df.shape)

df = df.dropna(subset=['price']).reset_index(drop=True)
median_price = df["price"].median()
df["price_label"] = (df["price"] > median_price).astype(int)

features = [c for c in df.columns if c not in ['model', 'price', 'price_label']]
X = df[features]
y = df["price_label"]

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='constant', fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

rf = RandomForestClassifier(n_estimators=200, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

pipe_rf = Pipeline([("pre", preprocessor), ("clf", rf)])
pipe_lr = Pipeline([("pre", preprocessor), ("clf", lr)])
pipe_gb = Pipeline([("pre", preprocessor), ("clf", gb)])

voting = VotingClassifier(
    estimators=[("rf", pipe_rf), ("gb", pipe_gb)], voting="soft"
)

voting.fit(X_train, y_train)
pred = voting.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

joblib.dump(voting, "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

summary = {
    "accuracy": float(accuracy),
    "features": features,
    "num_columns": num_cols,
    "categorical_columns": cat_cols
}
with open("training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Training completed and model saved.")
