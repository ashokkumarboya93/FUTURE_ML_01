#Here we import libraries like pandas , numpy and  sklearn also StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Here we can Load spotify dataset
spotify_data = "/content/dataset.csv"  #this line indicates spotify dataset file path
df = pd.read_csv(spotify_data)

# Display first 5 rows to check data
print("The Spotify songs Dataset Sample:")
print(df.head())

#Now removing the unnecessary columns from data
remove_unnce_cols = ['Unnamed: 0', 'track_id', 'album_name', 'track_name', 'artists']
df = df.drop(columns=[col for col in remove_unnce_cols if col in df.columns], errors='ignore') # Changed cremove_unnce_cols to remove_unnce_cols

# Handle missing values
df = df.dropna()

# Defining the  mood categories based on valence & energy
def classify_mood(valence, energy):
    if valence >= 0.5 and energy >= 0.5:
        return 'Happy'
    elif valence < 0.5 and energy >= 0.5:
        return 'Energetic'
    elif valence < 0.5 and energy < 0.5:
        return 'Sad'
    else:
        return 'Calm'

# Apply function to create 'mood' column and add to preprocessed dataset.
df['mood'] = df.apply(lambda row: classify_mood(row['valence'], row['energy']), axis=1)

# Selecting the features for training
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]
y = df['mood']

# Standardize features (important for ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converting  back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Save preprocessed dataset
df_processed = pd.concat([X_scaled_df, y], axis=1)
df_processed.to_csv("processed_spotify_data.csv", index=False)

print("\n The Data Preprocessing Done")
print(" Preprocessed dataset saved as 'processed_spotify_data.csv'.")
print("\n Mood Distribution:\n", df['mood'].value_counts())