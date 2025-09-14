import os, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_rf(X, y, out_dir: str, logger=print):
    os.makedirs(out_dir, exist_ok=True)
    le = LabelEncoder()
    
    y_enc = le.fit_transform(y)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_tr, y_tr)
    
    logger("\n" + classification_report(y_va, clf.predict(X_va), target_names=list(le.classes_)))
    model_path = os.path.join(out_dir, "gesture_classifier.pkl")
    le_path    = os.path.join(out_dir, "label_encoder.pkl")
    
    with open(model_path, "wb") as f: pickle.dump(clf, f)
    with open(le_path, "wb") as f: pickle.dump(le, f)
    return model_path, le_path