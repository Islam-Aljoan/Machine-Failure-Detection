# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop

sns.set_style('darkgrid')

# Load the dataset
df = pd.read_csv(r"C:\Users\islam\Desktop\LAST semester at GJU\Machine II\Assignment 4\archive\machine_failure_cleaned.csv")
df = df.rename(columns={'Machine failure': "IsFail"})

# Separate features and target
features = df[['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
labels = df['IsFail']

# Split the dataset into the train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Apply standard scaling to the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameters to tune
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [10, 20, 30]
num_epochs = [100, 150, 200]

best_precision = 0
best_recall = 0
best_params = {}

# Loop over hyperparameters
for lr in learning_rates:
    for batch_size in batch_sizes:
        for epochs in num_epochs:
            print(f"Training with learning rate={lr}, batch size={batch_size}, epochs={epochs}")

            # Build the ANN model
            classifier = Sequential()

            # First hidden layer
            classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=3))
            classifier.add(BatchNormalization())
            

            # Second hidden layer
            classifier.add(Dense(units=4, kernel_initializer="uniform", activation="relu"))
            classifier.add(BatchNormalization())
            classifier.add(Dropout(0.2))

            # Output layer
            classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

            # Compile the ANN with RMSprop optimizer
            optimizer = RMSprop(learning_rate=lr)
            classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

            # Train the ANN
            history = classifier.fit(X_train_resampled, y_train_resampled, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

            # Make predictions
            Y_predict = classifier.predict(X_test)
            Y_predict = (Y_predict > 0.5)

            # Confusion matrix and additional metrics
            precision = precision_score(y_test, Y_predict)
            recall = recall_score(y_test, Y_predict)

            if precision > best_precision and recall > best_recall:
                best_precision = precision
                best_recall = recall
                best_params = {'learning_rate': lr, 'batch_size': batch_size, 'epochs': epochs}

print("Best hyperparameters:")
print(best_params)
print(f"Best Precision: {best_precision:.2f}")
print(f"Best Recall: {best_recall:.2f}")

