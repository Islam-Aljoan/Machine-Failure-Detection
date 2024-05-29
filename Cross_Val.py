import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

sns.set_style('darkgrid')

# Load the dataset
df = pd.read_csv(r"C:\Users\islam\Desktop\LAST semester at GJU\Machine II\Assignment 4\archive\machine_failure_cleaned.csv")
df = df.rename(columns={'Machine failure': "IsFail"})

# Separate features and target
features = df[['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
labels = df['IsFail']

# Apply standard scaling to the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

precision_scores = []
recall_scores = []
f1_scores = []

# Cross-validation process
for train_index, test_index in kf.split(features_scaled):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Build the ANN model
    classifier = Sequential()
    
    # First hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=3))
    
    # Second hidden layer with dropout
    classifier.add(Dense(units=4, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dropout(0.2))
    
    # Output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    # Compile the ANN
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    
    # Train the ANN
    classifier.fit(X_train_resampled, y_train_resampled, batch_size=10, epochs=100, verbose=0)
    
    # Make predictions
    Y_predict = classifier.predict(X_test)
    Y_predict = (Y_predict > 0.5)
    
    # Confusion matrix and additional metrics
    precision = precision_score(y_test, Y_predict)
    recall = recall_score(y_test, Y_predict)
    f1 = f1_score(y_test, Y_predict)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Calculate average metrics
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)


print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1 Score: {avg_f1:.2f}")


