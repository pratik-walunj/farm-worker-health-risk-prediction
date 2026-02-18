import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
import seaborn as sns
import shap
import xgboost as xgb

from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# -------------------------------------------------
# 1️⃣ Compute Normalized Health Risk Index
# -------------------------------------------------

def compute_health_risk_index(df):

    df["Temp_N"] = (df["Temperature"] - 28) / (45 - 28)
    df["Humidity_N"] = (df["Humidity"] - 35) / (85 - 35)
    df["HR_N"] = (df["Heart_Rate"] - 60) / (130 - 60)
    df["Work_N"] = (df["Working_Hours"] - 4) / (12 - 4)

    df["Health_Risk_Index"] = (
        0.30 * df["Temp_N"] +
        0.20 * df["Humidity_N"] +
        0.30 * df["HR_N"] +
        0.20 * df["Work_N"]
    )

    return df


# -------------------------------------------------
# 2️⃣ Correlation Analysis
# -------------------------------------------------

def correlation_analysis(df):
    corr, p = pearsonr(df["Temperature"], df["Heart_Rate"])
    print(f"\nCorrelation between Temperature and Heart Rate: {corr}")
    print(f"p-value: {p}")


# -------------------------------------------------
# 3️⃣ Add Risk Category
# -------------------------------------------------

def add_risk_category(df):

    def categorize(hri):
        if hri < 0.4:
            return 0
        elif hri < 0.7:
            return 1
        else:
            return 2

    df["Risk_Category"] = df["Health_Risk_Index"].apply(categorize)
    return df


# -------------------------------------------------
# 4️⃣ Advanced Train ML Model (SMOTE + Tuning + CV)
# -------------------------------------------------

def train_ml_model(df):

    X = df[["Temperature", "Humidity", "Heart_Rate", "Working_Hours"]]
    y = df["Risk_Category"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nBest Parameters Found:", grid_search.best_params_)

    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return best_model, X.columns, X_test, y_test, acc


# -------------------------------------------------
# 5️⃣ Model Comparison (RF vs XGBoost)
# -------------------------------------------------

def compare_models(df):

    X = df[["Temperature", "Humidity", "Heart_Rate", "Working_Hours"]]
    y = df["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

    print("\n----- Model Comparison -----")
    print("Random Forest Accuracy:", rf_acc)
    print("XGBoost Accuracy:", xgb_acc)

    if xgb_acc > rf_acc:
        print("👉 XGBoost performs better.")
        return xgb_model, X.columns, X_test
    else:
        print("👉 Random Forest performs better.")
        return rf, X.columns, X_test


# -------------------------------------------------
# 6️⃣ SHAP Explainability
# -------------------------------------------------

def shap_explainability(model, X_test):

    print("\nGenerating SHAP Explainability...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance")
    plt.show()


# -------------------------------------------------
# 7️⃣ Feature Importance
# -------------------------------------------------

def plot_feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure()
        plt.bar(feature_names, importances)
        plt.xticks(rotation=45)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()


# -------------------------------------------------
# 8️⃣ Confusion Matrix
# -------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# -------------------------------------------------
# 9️⃣ ROC Curve
# -------------------------------------------------

def plot_roc_curve(model, X_test, y_test):

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    plt.figure()

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC = {roc_auc:.2f}")

    plt.legend()
    plt.title("ROC Curve")
    plt.show()


# -------------------------------------------------
# 🔟 Save / Load Model
# -------------------------------------------------

def save_model(model):
    joblib.dump(model, "model.pkl")
    print("Model saved successfully!")


def load_saved_model():
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
    return model


# -------------------------------------------------
# 1️⃣1️⃣ Real-Time Prediction
# -------------------------------------------------

def simulate_real_time_prediction(model):

    print("\n--- Real-Time Worker Health Prediction ---")

    try:
        temperature = float(input("Enter Temperature: "))
        humidity = float(input("Enter Humidity: "))
        heart_rate = float(input("Enter Heart Rate: "))
        working_hours = float(input("Enter Working Hours: "))

        input_df = pd.DataFrame(
            [[temperature, humidity, heart_rate, working_hours]],
            columns=["Temperature", "Humidity", "Heart_Rate", "Working_Hours"]
        )

        prediction = model.predict(input_df)[0]
        risk_labels = ["LOW", "MEDIUM", "HIGH"]

        print("Predicted Risk:", risk_labels[prediction])

    except:
        print("Invalid input.")


# -------------------------------------------------
# 1️⃣2️⃣ Live Sensor Simulation
# -------------------------------------------------

def live_sensor_simulation(model):

    print("\n--- Live Sensor Simulation ---")

    for _ in range(5):

        temperature = random.uniform(28, 45)
        humidity = random.uniform(35, 85)
        heart_rate = random.uniform(60, 130)
        working_hours = random.uniform(4, 12)

        input_df = pd.DataFrame(
            [[temperature, humidity, heart_rate, working_hours]],
            columns=["Temperature", "Humidity", "Heart_Rate", "Working_Hours"]
        )

        prediction = model.predict(input_df)[0]
        risk_labels = ["LOW", "MEDIUM", "HIGH"]

        print(f"\nTemp: {temperature:.2f}, Humidity: {humidity:.2f}, "
              f"HR: {heart_rate:.2f}, Hours: {working_hours:.2f}")
        print("Predicted Risk:", risk_labels[prediction])


# -------------------------------------------------
# 1️⃣3️⃣ PDF Report
# -------------------------------------------------

def generate_pdf_report(accuracy):

    doc = SimpleDocTemplate("Health_Risk_Report.pdf", pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Farm Worker Health Risk Analysis Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Model Accuracy: {accuracy:.4f}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [
        ["Risk Level", "Description"],
        ["Low", "Safe condition"],
        ["Medium", "Monitor closely"],
        ["High", "Immediate intervention required"]
    ]

    elements.append(Table(table_data))
    doc.build(elements)

    print("PDF Report Generated Successfully!")
