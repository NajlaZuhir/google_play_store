import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("new_df.csv")

# Drop columns not used in the model
X = df.drop(columns=['App', 'Success Category', 'Discrepancy', 'User_Engagement'])

# Ensure all categorical data is encoded
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables

# Save feature names after encoding
feature_names = X.columns.tolist()
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Define the target variable
y = df["Success Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hybrid sampling strategy (SMOTE + ENN)
hybrid_sampler = SMOTEENN(random_state=42)

# Apply hybrid sampling
X_train_resampled, y_train_resampled = hybrid_sampler.fit_resample(X_train, y_train)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and save each model
for model_name, model in models.items():
    print(f"Training {model_name} with SMOTEENN...")

    # Train the model
    model.fit(X_train_resampled, y_train_resampled)

    # Save the trained model
    filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    pickle.dump(model, open(filename, "wb"))
    print(f"{model_name} saved as {filename}")

print("\nModels have been trained and saved successfully!")
