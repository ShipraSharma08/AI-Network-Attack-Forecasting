import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = "data/processed/state_transitions_clean.csv"

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# TEMPORAL SPLIT
# ---------------------------------------------------------
# We preserve chronological order.
# This single-day event requires the training set to contain
# both benign and attack observations.
train_df = df[df["window"] <= 609].copy()
val_df = df[(df["window"] >= 610) & (df["window"] <= 629)].copy()
test_df = df[df["window"] >= 630].copy()

# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------
# ONLY current-state features are used.
# next_* features would leak future information.
feature_cols = [
    c for c in df.columns
    if c.startswith("current_")
    and c != "current_attack_label"
]

X_train = train_df[feature_cols]
y_train = train_df["next_attack_label"]

X_val = val_df[feature_cols]
y_val = val_df["next_attack_label"]

X_test = test_df[feature_cols]
y_test = test_df["next_attack_label"]

# Clean invalid numerical values
X_train = X_train.replace([float("inf"), float("-inf")], 0).fillna(0)
X_val = X_val.replace([float("inf"), float("-inf")], 0).fillna(0)
X_test = X_test.replace([float("inf"), float("-inf")], 0).fillna(0)

# ---------------------------------------------------------
# VALIDATE CLASS DISTRIBUTION
# ---------------------------------------------------------
print("\n===== TEMPORAL SPLIT =====")
print("Train:", len(train_df), "samples")
print("Validation:", len(val_df), "samples")
print("Test:", len(test_df), "samples")

print("\nTarget distribution:")
print("Train:")
print(y_train.value_counts().sort_index())

print("Validation:")
print(y_val.value_counts().sort_index())

print("Test:")
print(y_test.value_counts().sort_index())

# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------
# Fit scaler ONLY on training data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# TEST EVALUATION
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print("\n===== LOGISTIC REGRESSION BASELINE =====")
print(f"Features used      : {len(feature_cols)}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"False Positive Rate: {fpr:.4f}")

print("\nConfusion Matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
