# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

file_path = 'Heart Prediction Quantum Dataset.csv'
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Overview:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution (Heart Disease):")
print(df['HeartDisease'].value_counts())

# Separate features (X) and target (y)
X = df.drop('HeartDisease', axis=1)  # Features
y = df['HeartDisease']  #output

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# First split into train (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.4, stratify=y, random_state=42
)

# Split temp into validation (20%) and test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\nData Splitting Results:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Test different K values (1 to 30)
k_values = range(1, 31)
val_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_pred = knn.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_accuracies.append(val_acc)

# Plot K vs. Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, val_accuracies, marker='o')
plt.title("Validation Accuracy vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# Select best K
best_k = k_values[np.argmax(val_accuracies)]
print(f"\nOptimal K: {best_k}")

# Train final model with best K
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

# Evaluate on test set
test_pred = final_knn.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test Accuracy (K={best_k}): {test_acc:.4f}")


# Cross-Validation (5-Fold)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_knn, X_train, y_train, cv=cv, scoring='accuracy')

print("\nCross-Validation Results:")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Std Dev: {np.std(cv_scores):.4f}")

# Compare performance metrics
train_acc = final_knn.score(X_train, y_train)
val_acc = accuracy_score(y_val, final_knn.predict(X_val))

print("\nPerformance Comparison:")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# Confusion Matrix & Metrics
cm = confusion_matrix(y_test, test_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate metrics
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)

print("\nClassification Metrics:")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Overfitting Analysis & Improvements
print("\nOverfitting Analysis:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Difference: {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.1:
    print("⚠️ Warning: Potential overfitting!")
else:
    print("✅ No significant overfitting detected.")

# Optional: Feature reduction using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_train_pca, y_train)
test_acc_pca = knn_pca.score(X_test_pca, y_test)

print(f"\nTest Accuracy after PCA (2D): {test_acc_pca:.4f}")

# Plot PCA components (if meaningful)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], label='No Disease', alpha=0.7)
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], label='Disease', alpha=0.7)
plt.title("2D PCA Projection of Heart Disease Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()