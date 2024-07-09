import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the data
with open('../data/train_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('../data/test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Initialize classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": MultinomialNB(),
    "GradientBoosting": GradientBoostingClassifier()
}

# Train and evaluate each classifier
best_accuracy = 0
best_model = None

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

# Save the best model
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best Model: {best_model.__class__.__name__} with accuracy {best_accuracy}")
