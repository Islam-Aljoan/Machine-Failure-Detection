# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('darkgrid')


df=pd.read_csv(r"C:\Users\islam\Desktop\LAST semester at GJU\Machine II\Assignment 4\archive\machine_failure_cleaned.csv")
df.head()
#df.info()

# Rename the Target column to IsFail
df = df.rename(columns = {'Machine failure': "IsFail"})
#df.info()

# Create the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Plot a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True)
# plt.show()


# Plot histograms of select features
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Change to 1 row, 3 columns
# columns = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# data = df.copy()

# for ind, item in enumerate(columns):
#     column = columns[ind]
#     df_column = data[column]
#     df_column.hist(ax = axes[ind], bins=32).set_title(item)  # Adjusted index

# fig.supylabel('count')
# fig.subplots_adjust(wspace=0.3)  # Adjusted horizontal spacing
# plt.show()


# Assuming df is your DataFrame and 'IsFail' is the column indicating failure status
# Example: df = pd.read_csv('your_dataset.csv')

# Assuming df is your DataFrame and 'IsFail' is the column indicating failure status
# Example: df = pd.read_csv('your_dataset.csv')

# Count the number of samples for each class
class_counts = df['IsFail'].value_counts()

# Extract the counts
num_unfailed = class_counts[0]  # Number of unfailed machines (IsFail=0)
num_failed = class_counts[1]  # Number of failed machines (IsFail=1)

# Plot the histogram
# labels = ['Unfailed Machines', 'Failed Machines']
# counts = [num_unfailed, num_failed]

# print(f"Number of failed machines (IsFail=1): {num_failed}")
# print(f"Number of unfailed machines (IsFail=0): {num_unfailed}")

# plt.figure(figsize=(8, 6))
# plt.bar(labels, counts, color=['blue', 'red'])
# plt.xlabel('Machine Status')
# plt.ylabel('Number of Samples')
# plt.title('Number of Failed and Unfailed Machines')
# plt.show()

# Separate features and target
features = df[['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
labels = df['IsFail']

# Split the dataset into the train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


from imblearn.over_sampling import SMOTE

# Assuming 'IsFail' is the target column and the rest are features
X = df.drop('IsFail', axis=1)
y = df['IsFail']

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Count the number of samples for each class before and after SMOTE
class_counts_original = y_train.value_counts()
class_counts_resampled = y_train_resampled.value_counts()

# print("Original class distribution:")
# print(class_counts_original)

# print("Resampled class distribution:")
# print(class_counts_resampled)

# Plotting the original class distribution
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.bar(class_counts_original.index, class_counts_original.values, color=['blue', 'red'])
# plt.xlabel('Machine Status')
# plt.ylabel('Number of Samples')
# plt.title('Original Class Distribution')
# plt.xticks([0, 1], ['Unfailed Machines', 'Failed Machines'])

# Plotting the resampled class distribution
# plt.subplot(1, 2, 2)
# plt.bar(class_counts_resampled.index, class_counts_resampled.values, color=['blue', 'red'])
# plt.xlabel('Machine Status')
# plt.ylabel('Number of Samples')
# plt.title('Resampled Class Distribution')
# plt.xticks([0, 1], ['Unfailed Machines', 'Failed Machines'])

# plt.tight_layout()
# plt.show()


import keras
#import 2 modules
#sequential module is to initialize neural network
#dense module to build layers neural network
from keras.models import Sequential
from keras.layers import Dense

#initialize NN using sequential, defining sequence of layer
classifier = Sequential() #we use classification because were going to predict Isfail
#using classifier

classifier.add(Dense(
		units = 6,
		kernel_initializer="uniform",
		activation="relu",
		input_dim = 3
		))

#next step is add second hidden layer
classifier.add(Dense(
		units = 4,
		kernel_initializer="uniform",
		activation="relu"
		))

classifier.add(Dense(
		units = 1,
		kernel_initializer="uniform",
		activation="sigmoid"
		))

#compiling ANN
classifier.compile(
		optimizer = "adam",
		loss="binary_crossentropy",
		metrics=['accuracy']
		)

classifier.fit(
		X_train_resampled,
		y_train_resampled,
		batch_size=10,
		epochs=100)

#part 5 is we gonna make an prediction with classifier

Y_predict = classifier.predict(X_test)

#after we get the predict value, convert it into boolean with threshold
Y_predict = (Y_predict > 0.5)

#after we make an prediction, we need make a confusion matrix to find the accuracy of testing data

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_predict)


