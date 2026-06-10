import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset

import pandas as pd
df = pd.read_csv("BreastCancer.csv")
# Remove unnecessary columns
df.drop(columns=["id"], inplace=True)

# Convert diagnosis
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Features
X = df.drop("diagnosis", axis=1)

# Target
y = df["diagnosis"]

print("Dataset Shape:", df.shape)
print("Number of Features:", X.shape[1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

# Accuracy
knn_accuracy = accuracy_score(y_test, predictions)
print("KNN Accuracy:", round(knn_accuracy, 4))
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(classification_report(y_test, predictions))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_predictions = nb.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print("\nModel Comparison")
print("----------------")

print("Naive Bayes Accuracy:", round(nb_accuracy, 4))
print("KNN Accuracy:", round(knn_accuracy, 4))
from sklearn.linear_model import Perceptron

perceptron = Perceptron(random_state=42)

perceptron.fit(X_train, y_train)

per_predictions = perceptron.predict(X_test)

per_accuracy = accuracy_score(y_test, per_predictions)

print("Perceptron Accuracy:", round(per_accuracy, 4))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)

clusters = kmeans.fit_predict(X)

print("\nK-Means clustering completed.")

results = {
    "KNN": knn_accuracy,
    "Naive Bayes": nb_accuracy,
    "Perceptron": per_accuracy
}

best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", round(results[best_model], 4))

models = ["KNN", "Naive Bayes", "Perceptron"]

accuracies = [
    knn_accuracy,
    nb_accuracy,
    per_accuracy
]

plt.bar(models, accuracies)

plt.title("Machine Learning Model Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()
correlation = df.corr()

plt.figure(figsize=(12,10))

plt.imshow(correlation)

plt.colorbar()

plt.title("Feature Correlation Matrix")

plt.savefig("correlation_matrix.png")
plt.show()

correlation_with_target = df.corr()["diagnosis"]

print("\nTop Features Related to Diagnosis:")

print(
    correlation_with_target
    .abs()
    .sort_values(ascending=False)
    .head(10)
)