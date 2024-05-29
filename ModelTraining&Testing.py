import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import itertools

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42, stratify=labels)

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
history = classifier.fit(X_train_resampled, y_train_resampled, batch_size=10, epochs=100, verbose=0, validation_data=(X_test, y_test))

# Make predictions
Y_predict = classifier.predict(X_test)
Y_predict_binary = (Y_predict > 0.5).astype(int)

# Calculate precision, recall, and F1 score

precision_micro = precision_score(y_test, Y_predict_binary, average='micro')
recall_micro = recall_score(y_test, Y_predict_binary, average='micro')

precision_weighted = precision_score(y_test, Y_predict_binary, average='weighted')
recall_weighted = recall_score(y_test, Y_predict_binary, average='weighted')

# Print the metrics
print(f"Micro-Average Precision: {precision_micro:.2f}")
print(f"Micro-Average Recall: {recall_micro:.2f}")
print(f"Weighted-Average Precision: {precision_weighted:.2f}")
print(f"Weighted-Average Recall: {recall_weighted:.2f}")


# Calculate and plot the confusion matrix
cm = confusion_matrix(y_test, Y_predict_binary)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['Not Fail', 'Fail'], title='Confusion Matrix')
plt.show()

# Plot ROC curves
fpr_micro, tpr_micro, _ = roc_curve(y_test, Y_predict)
roc_auc_micro = roc_auc_score(y_test, Y_predict)

fpr_weighted, tpr_weighted, _ = roc_curve(y_test, Y_predict, sample_weight=(y_test * precision_weighted))
roc_auc_weighted = roc_auc_score(y_test, Y_predict, sample_weight=(y_test * precision_weighted))

plt.figure()
plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=2, label=f'ROC curve (micro-average) (area = {roc_auc_micro:.2f})')
plt.plot(fpr_weighted, tpr_weighted, color='blue', lw=2, label=f'ROC curve (weighted-average) (area = {roc_auc_weighted:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot loss function vs. epochs
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function vs. Epochs')
plt.legend()
plt.show()
