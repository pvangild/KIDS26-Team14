import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report

train_windows = pd.read_csv("data/train_windows.csv")
val_windows = pd.read_csv("data/val_windows.csv")

feature_cols = [col for col in train_windows.columns if col not in ["study_number", "label"]]

X_train = train_windows[feature_cols]
y_train = train_windows["label"]
X_val = val_windows[feature_cols]
y_val = val_windows["label"]

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    class_weight={"normal": 1, "apnea": 5, "hypopnea": 5},
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

probabilities = model.predict_proba(X_val)
classes = list(model.classes_)

val_windows["prob_apnea"] = probabilities[:, classes.index("apnea")]
val_windows["prob_hypopnea"] = probabilities[:, classes.index("hypopnea")]
val_windows["prob_normal"] = probabilities[:, classes.index("normal")]

print(classification_report(y_val, y_pred))
val_windows["predicted_label"] = y_pred
val_windows.to_csv("data/val_predictions.csv", index=False)