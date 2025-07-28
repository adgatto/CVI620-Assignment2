import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')


# DATA

def load_data():
    data_list = []
    labels = []

    for i, address in enumerate(glob.glob('train/*/*')):
        img = cv2.imread(address)
        img = cv2.resize(img, (32, 32)) 
        img = img / 255.0    
        img = img.flatten()
        data_list.append(img)

        # Get the label of either cat or dog
        label = address.split("\\")[-2] if "\\" in address else address.split("/")[-2]
        labels.append(label)

        if i % 500 == 0:
            print(f"[INFO]: {i} images processed")

    data_list = np.array(data_list)
    labels = np.array(labels)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data_list, labels, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Load the data
X_train, X_test, y_train, y_test = load_data()

# MODEL
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Perceptron': Perceptron(max_iter=2000, tol=1e-3),
}

# RESULTS AND EVALUATION

# Train and evaluate all models, keep track of the best one
best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {acc:.4f}")

    # Track best model
    if acc > best_acc:
        best_acc = acc
        best_model = model

# SAVE MODEL
# Based on the best accuracy, Ill be using the Logistic Regression model.
# K-Nearest Neighbors Accuracy: 0.5650
# Logistic Regression Accuracy: 0.5925
# Perceptron Accuracy: 0.5500
dump(best_model, 'cat_dog_classifier.joblib')
