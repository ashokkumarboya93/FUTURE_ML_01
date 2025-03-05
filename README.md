Task 1: Classify Spotify Songs by Mood :
-----------------------------------------

Overview :
----------
This task focuses on classifying Spotify songs into different moods (Happy, Sad, Energetic, Calm) using machine learning techniques. It also includes data preprocessing, feature extraction, and visualization.

Skills Gained :
----------------
- Classification
- Data Preprocessing
- Audio Feature Analysis
- Data Visualization

Dataset :
----------
-> File: `processed_spotify_data.csv`
-> Source:Preprocessed Spotify song dataset
-> Features: Audio features like valence, energy, danceability, etc.
-> Target: Mood classification (Happy, Sad, Energetic, Calm)

Tools & Libraries Used :
--------------------------
-> Python
-> Pandas, NumPy (Data Handling)
-> Matplotlib, Seaborn (Visualization)
-> Scikit-learn (Machine Learning)
-> Librosa (Audio Feature Extraction)

 Implementation Steps :
 ----------------------
1. Data Preprocessing :
   -> Load and clean dataset (`processed_spotify_data.csv`)
   -> Handle missing values and normalize features
2. Feature Extraction :
   -> Extract relevant audio features
3. Train Classification Model :
   -> Model: `RandomForestClassifier`
   -> Optimized with hyperparameter tuning
4. Model Evaluation :
   -> Accuracy Calculation
   -> Generate Confusion Matrix & Classification Report
5. Data Visualization :
   -> Mood Distribution (Bar & Pie Chart)
   -> Valence vs Energy Scatter Plot
   -> Feature Histograms
   -> Playlist Classification Counts
   -> Confusion Matrix Heatmap

Results :
----------
Model Accuracy: 91%(0.91)
Classification Report:
-------------------------------------------------------
              precision    recall  f1-score   support
        Calm       0.86      0.66      0.74      1750
   Energetic       0.91      0.93      0.92      7752
       Happy       0.92      0.95      0.94      8681
         Sad       0.90      0.90      0.90      4617

    accuracy                           0.91     22800
   macro avg       0.90      0.86      0.87     22800
weighted avg       0.91      0.91      0.91     22800

--------------------------------------------------------
Confusion Matrix:
-----------------------

[[1151   39  395  165]
 [   9 7174  274  295]
 [ 138  299 8234   10]
 [  44  400   19 4154]]

------------------------

How to Run the Code :
----------------------
1. Clone the Repository:
   bash :
   git clone https://github.com/yourusername/spotify-mood-classification.git
   cd spotify-mood-classification
   
2. Install Dependencies:
   bash :
   pip install pandas numpy matplotlib seaborn scikit-learn librosa
   
3. Run the Python Script:
   bash :
   python mood_classification.py
   
4. View the Plots & Model Results.


Contact :
----------
For any queries, reach out via GitHub Issues or email.

Happy Mood Classification! Thank you for visiting.

Internship Details :
---------------------
Offered By: Future Interns
Role: Machine Learning Intern
Task 01: Spotify Song Mood Classification

