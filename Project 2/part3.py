import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from sklearn.metrics import confusion_matrix


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(f"{filename}")
    return data["features"], data["domains"], data["digits"]


features_train, domains_train, digits_train = load_data("train_data.npz")
features_test, domains_test, digits_test = load_data("test_data.npz")

rf = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=50)

rf.fit(features_train, domains_train)
predicted_domains_test = rf.predict(features_test)

cm_initial = confusion_matrix(domains_test, predicted_domains_test)
print("Confusion matrix (before):")
print(cm_initial)


X_domain2 = features_test[domains_test == 3]
X_domain3 = features_test[domains_test == 4]

weight = 0.076


num_samples_add = int(len(X_domain2) * weight)


random_indices = np.random.choice(len(X_domain2), size=num_samples_add, replace=False)
selected_X_domain2 = X_domain2[random_indices]

X_domain3 = np.concatenate((X_domain3, selected_X_domain2), axis=0)


features_test_updated = np.concatenate((features_test[domains_test != 4], X_domain3))


predicted_domains_updated_test = rf.predict(features_test_updated)


cm_updated = confusion_matrix(
    domains_test, predicted_domains_updated_test[: len(domains_test)]
)
print("Confusion matrix (after):")
print(cm_updated)

unique_domains = np.unique(predicted_domains_updated_test[: len(domains_test)])

for domain in unique_domains:
    domain_indices = np.where(
        predicted_domains_updated_test[: len(domains_test)] == domain
    )[0]
    domain_features_test = features_test[domain_indices]
    domain_digits_test = digits_test[domain_indices]

    rf_domain = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=50)
    rf_domain.fit(features_train, digits_train)

    domain_test_score = rf_domain.score(domain_features_test, domain_digits_test)
    print(f"Accuracy for domain {domain}: {domain_test_score}")

    cm_domain_test = confusion_matrix(
        domain_digits_test, rf_domain.predict(domain_features_test)
    )
    print(f"Confusion matrix for domain {domain}:")
    print(cm_domain_test)
