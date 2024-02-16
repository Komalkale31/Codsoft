import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

over_sampler = RandomOverSampler(sampling_strategy=0.5)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

under_sampler = RandomUnderSampler(sampling_strategy=0.5)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_under, y_train_under)

lr_classifier = LogisticRegression(random_state=42)
lr_classifier.fit(X_train_under, y_train_under)

y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

y_pred_lr = lr_classifier.predict(X_test)
print("Logistic Regression Classifier:")
print(classification_report(y_test, y_pred_lr))
