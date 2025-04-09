import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load the dataset
career = pd.read_csv('dataset9000.data', header=None)

# Define column names
career.columns = ["Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
"Cyber Security", "Networking", "Development", "Programming Skills", "Project Management",
"Computer Forensics Fundamentals", "Technical Communication", "AI ML", "Software Engineering", "Business Analysis",
"Communication skills", "Data Science", "Troubleshooting skills", "Graphics Designing", "Roles"]

# Drop rows with all missing values
career.dropna(how='all', inplace=True)

# Convert non-numeric columns to numeric using LabelEncoder
label_encoder = LabelEncoder()
career['Roles'] = label_encoder.fit_transform(career['Roles'])  # Encode target column

# Ensure all features are numeric
X = np.array(career.iloc[:, 0:17], dtype=float)  # Convert features to float
y = np.array(career.iloc[:, 17])  # Target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=524)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = knn.predict(X_test)
scores = {}
scores[5] = metrics.accuracy_score(y_test, y_pred)
print('Accuracy =', scores[5] * 100)

# Save the model to a file
pickle.dump(knn, open('careerlast.pkl', 'wb'))
print('Model saved as careerlast.pkl')