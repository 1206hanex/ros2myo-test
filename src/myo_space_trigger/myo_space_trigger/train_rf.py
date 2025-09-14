import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load training data
df = pd.read_csv("training_data_fe.csv")
X = df.drop(columns=["label"])
y = df["label"]

# train/test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# save model to .pkl file
with open("gesture_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Model saved to gesture_classifier.pkl")

# evaluate
y_pred = clf.predict(X_val)
print("Evaluation on validation set:")
print(classification_report(y_val, y_pred))
