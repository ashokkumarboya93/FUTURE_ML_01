import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load Dataset
df = pd.read_csv("/content/processed_spotify_data.csv")

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'track_id', 'album_name', 'track_name', 'artists']
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Check for highly correlated features
numerical_features = df.select_dtypes(include=np.number).columns
corr_matrix = df[numerical_features].corr()
high_corr_features = {corr_matrix.columns[i] for i in range(len(corr_matrix.columns)) 
                      for j in range(i) if abs(corr_matrix.iloc[i, j]) > 0.85}
df.drop(columns=high_corr_features, errors='ignore', inplace=True)

# Define features and target
if 'mood' not in df.columns:
    raise KeyError("The 'mood' column is not found in the DataFrame. Please check your data.")

X = df.drop(columns=['mood'])
y = df['mood']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (Preserve 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Results
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Count of Songs in Each Playlist
playlist_song_counts = df['mood'].value_counts()
print("\nCount of Songs in Each Playlist:")
print(playlist_song_counts)

# Define class labels
class_labels = y.unique()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve Calculation
roc_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average='macro')
fpr, tpr, _ = roc_curve(pd.get_dummies(y_test).values.ravel(), pd.get_dummies(y_pred).values.ravel())

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Visualization: Mood Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y_test, palette='viridis')
plt.title("Mood Distribution in Dataset")
plt.show()
