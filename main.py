import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
file_path = os.path.join("data", "churn_data.csv")

df = pd.read_csv(file_path, encoding='latin1', engine='python')

print("✅ Dataset Loaded Successfully")
print(df.head())

# -------------------------------
# 2. DATA CLEANING
# -------------------------------
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df.dropna(inplace=True)

# -------------------------------
# 3. EDA VISUALS
# -------------------------------
os.makedirs("outputs", exist_ok=True)

plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.close()

plt.figure()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=30)
plt.savefig("outputs/churn_contract.png")
plt.close()

plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("outputs/monthly_charges.png")
plt.close()

plt.figure()
sns.histplot(data=df, x="tenure", hue="Churn", bins=30)
plt.title("Tenure vs Churn")
plt.savefig("outputs/tenure.png")
plt.close()

# -------------------------------
# 4. ENCODING
# -------------------------------
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 5. CORRELATION HEATMAP
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")
plt.close()

# -------------------------------
# 6. SPLIT DATA
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. TRAIN MODEL
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# 8. PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9. EVALUATION
# -------------------------------
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# -------------------------------
# 10. FEATURE IMPORTANCE
# -------------------------------
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

# -------------------------------
# 11. SAVE MODEL (FIXED)
# -------------------------------
os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "churn_model.pkl")

joblib.dump(model, model_path)

print("\n💾 Model saved at:", model_path)
print("📁 Outputs saved in 'outputs/' folder")
print("🚀 Project completed successfully!")