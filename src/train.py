import pandas as pd

# Build the expected CMAPSS FD001 schema: id/cycle, 3 operating settings, 21 sensors.
def get_column_names():
    columns = ["engine_id", "cycle"]
    columns += [f"op_setting_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    return columns

# Read the raw training text file (space-separated) and assign stable column names.
def load_train_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Keep only the first 26 usable columns in case of trailing empty tokens.
    df = df.iloc[:, :26]
    df.columns = get_column_names()
    return df

# Compute Remaining Useful Life (RUL) per row as max engine cycle minus current cycle.
def add_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop(columns=["max_cycle"], inplace=True)
    return df

# Load and enrich the dataset before creating labels.
df = load_train_data("train_FD001.txt")
df = add_rul(df)

print(df.head())
print("\nShape:", df.shape)
print("\nRUL stats:\n", df["RUL"].describe())


def create_health_label(rul):
    # Bucketize RUL into 3 health classes for classification.
    if rul > 60:
        return 2   # Healthy
    elif rul > 30:
        return 1   # Warning
    else:
        return 0   # Critical

# Create categorical target from the continuous RUL target.
df["label"] = df["RUL"].apply(create_health_label)

print(df[["engine_id", "cycle", "RUL", "label"]].head(10))
print("\nClass distribution:\n", df["label"].value_counts().sort_index())


# Exclude identifiers and targets to build the model input matrix.
feature_cols = [col for col in df.columns if col not in ["engine_id", "cycle", "RUL", "label"]]

X = df[feature_cols]
y = df["label"]

print("Number of features:", len(feature_cols))
print("X shape:", X.shape)
print("y shape:", y.shape)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Stratified split preserves class ratios across train/validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a baseline ensemble classifier on tabular sensor features.
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Report core classification metrics to inspect overall and per-class performance.
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=["Critical", "Warning", "Healthy"]))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))


