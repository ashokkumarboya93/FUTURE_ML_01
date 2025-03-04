import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("/content/processed_spotify_data.csv")

# Set the style
sns.set_style("darkgrid")

# Create a figure with subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 12))

# 🔹 1. Bar Chart - Mood Distribution
sns.barplot(x=df['mood'].value_counts().index,
            y=df['mood'].value_counts().values,
            palette="viridis", ax=axes[0]) # Changed axes[0, 0] to axes[0]


axes[0].set_title("Mood Distribution in Spotify Dataset")
axes[0].set_xlabel("Mood")
axes[0].set_ylabel("Count")


# 🔹 3. Scatter Plot - Valence vs Energy (Mood Separation)
sns.scatterplot(x=df['valence'], y=df['energy'], hue=df['mood'], palette="deep", alpha=0.7, ax=axes[1]) # Changed axes[1, 0] to axes[1]
axes[1].set_title("Mood Classification based on Valence & Energy")
axes[1].set_xlabel("Valence (Happiness)")
axes[1].set_ylabel("Energy (Liveliness)")


# Adjust layout
plt.tight_layout()
plt.show()
# Visualization: Feature Importance
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns[:len(feature_importances)], palette='coolwarm')
plt.title("Feature Importance")
plt.show()
# Feature Distribution Plots
plt.figure(figsize=(12, 6))
df.drop(columns=['mood']).hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Audio Feature Distributions", fontsize=16)
plt.show()