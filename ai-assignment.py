import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
# Adjust the delimiter and headers according to your file format
data = pd.read_csv('echocardiogram.data', delimiter=' ', header=None)  # Replace ' ' with actual delimiter

# Assign column names if the dataset doesn't have headers
# Replace 'Feature1', 'Feature2', etc., with your actual column names
# Ensure the last column is 'Emotion_Label' or equivalent
column_names = [f"Feature{i}" for i in range(1, data.shape[1])]  # Auto-generate feature names
data.columns = column_names[:-1] + ['Emotion_Label']

# Convert the emotion labels to categorical format
data['Emotion_Label'] = data['Emotion_Label'].astype('category').cat.codes  # Encode labels to numeric
emotion_labels = data['Emotion_Label'].unique()  # List of unique emotion classes
num_emotions = len(emotion_labels)  # Number of unique emotions
y = to_categorical(data['Emotion_Label'])  # One-hot encode the labels

# Extract features and labels
X = data.drop(columns=['Emotion_Label'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (if applicable)
pca = PCA(n_components=50)  # Adjust components based on your dataset
do_pca = True  # Set to False if PCA is not required
if do_pca:
    X_processed = pca.fit_transform(X_scaled)
else:
    X_processed = X_scaled

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(num_emotions, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    np.expand_dims(X_train, axis=-1), y_train,  # Add an extra dimension for Conv1D input
    validation_data=(np.expand_dims(X_test, axis=-1), y_test),
    epochs=50, batch_size=32, callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(np.expand_dims(X_test, axis=-1), y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions
predictions = model.predict(np.expand_dims(X_test, axis=-1))
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=[str(lbl) for lbl in emotion_labels]))

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
