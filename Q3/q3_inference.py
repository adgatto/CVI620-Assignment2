import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

# DATA
data = pd.read_csv('./mnist_test.csv', header=None)

# MODEL
model = load('mnist_best_model.joblib')
X_test = data.iloc[:, 1:].values / 255.0
y_test = data.iloc[:, 0].values.astype(int)

# RESULTS AND EVALUATION
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {acc:.2f}")
