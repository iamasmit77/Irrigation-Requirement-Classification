# ============================================================
# IRRIGATION REQUIREMENT CLASSIFICATION
# Learn Depth Machine Learning Internship - Track 1
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("irrigation_capstone_10000.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\nIrrigation Need Distribution:")
print(df["Irrigation_Need"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Irrigation_Need")
plt.title("Irrigation Need Distribution")
plt.xlabel("Irrigation Need")
plt.ylabel("Number of Records")
plt.show()


# ============================================================
# 5. REMOVE UNNECESSARY ID COLUMN
# ============================================================

df = df.drop("id", axis=1)

print("\nID column removed.")
print("New dataset shape:", df.shape)


# ============================================================
# 6. SELECT FEATURES AND TARGET
# ============================================================

X = df.drop("Irrigation_Need", axis=1)

y = df["Irrigation_Need"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 9. DATA PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 10. CREATE MODELS
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


decision_tree_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42,
            max_depth=8
        ))
    ]
)


knn_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]
)


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)


# ============================================================
# 12. TRAIN DECISION TREE
# ============================================================

print("\nTraining Decision Tree...")

decision_tree_model.fit(X_train, y_train)

decision_tree_prediction = decision_tree_model.predict(X_test)


# ============================================================
# 13. TRAIN KNN
# ============================================================

print("\nTraining KNN...")

knn_model.fit(X_train, y_train)

knn_prediction = knn_model.predict(X_test)


# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n================================")
    print(model_name)
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))

    return accuracy, precision, recall, f1


# ============================================================
# 15. EVALUATE ALL MODELS
# ============================================================

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_prediction
)

dt_results = evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_prediction
)

knn_results = evaluate_model(
    "KNN",
    y_test,
    knn_prediction
)


# ============================================================
# 16. COMPARE MODEL PERFORMANCE
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN"
    ],

    "Accuracy": [
        lr_results[0],
        dt_results[0],
        knn_results[0]
    ],

    "Precision": [
        lr_results[1],
        dt_results[1],
        knn_results[1]
    ],

    "Recall": [
        lr_results[2],
        dt_results[2],
        knn_results[2]
    ],

    "F1 Score": [
        lr_results[3],
        dt_results[3],
        knn_results[3]
    ]
})

print("\n\nMODEL COMPARISON")
print(results)


# ============================================================
# 17. CONFUSION MATRIX - DECISION TREE
# ============================================================

cm = confusion_matrix(
    y_test,
    decision_tree_prediction
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique())
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 18. SELECT MODEL
# ============================================================

best_model = decision_tree_model

print("\nSelected model: Decision Tree")


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "irrigation_model.pkl"
)

print("\nModel saved successfully!")
print("File created: irrigation_model.pkl")


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n================================")
print("PROJECT DEVELOPMENT COMPLETED")
print("================================")
print("Dataset:", df.shape)
print("Models trained: 3")
print("Model saved: irrigation_model.pkl")