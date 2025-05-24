# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

# === Step 1: Load training data ===
train_df = pd.read_csv("train.csv")  # Ensure train.csv is in the same directory or give full path

# === Step 2: Encode categorical variables ===
df = train_df.copy()

# Encode features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'y':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# === Step 3: Split data (optional, for testing) ===
X = df.drop(columns='y')
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Save the model ===
dump(model, 'term_deposit_model.joblib')
print("✅ Model saved as term_deposit_model.joblib")
