import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from sklearn.metrics import confusion_matrix


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(f"{filename}")
    return data["features"], data["domains"], data["digits"]


features_train, domains_train, digits_train = load_data("train_data.npz")
features_test, domains_test, digits_test = \
    load_data('test_data.npz')
    

# Create a random forest classifier model
rf = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=50)
rf.fit(features_train, domains_train)
train_score = rf.score(features_train, domains_train)
print("Training accuracy:", train_score)
predicted_domains_train = rf.predict(features_train)
cm_train = confusion_matrix(domains_train, predicted_domains_train)
print("Confusion matrix for training data:")
print(cm_train)