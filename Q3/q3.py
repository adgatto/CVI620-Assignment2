import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# DATA
data = pd.read_csv('./mnist_train.csv', header=None)  

# First column = label (y), 
# remaining columns = pixels (X)
X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values   

# Normalize pixel values to [0, 1]
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

# RESULTS AND EVALUATION
best_acc = 0
best_model = None

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"Best model: {best_model} with accuracy {best_acc:.2f}")

# Save best model
dump(best_model, 'mnist_best_model.joblib')
