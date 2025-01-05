import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv('echocardiogram.data', header=None, on_bad_lines='skip')

# Assign column names
data.columns = [
    "Instance_ID", "Survival_Status", "Age", "Pericardial_Effusion", 
    "Fractional_Shortening", "Wall_Motion_Score", "LVEDD", 
    "Ejection_Fraction", "Group", "Alive_One_Year", "Name", 
    "Label", "Another_Feature"
]

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Display dataset information
print("Dataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Drop irrelevant columns
data = data.drop(["Instance_ID", "Name", "Another_Feature", "Group"], axis=1)

# Convert numeric columns to numeric types
numeric_columns = [
    "Age", "Fractional_Shortening", "Wall_Motion_Score", 
    "LVEDD", "Ejection_Fraction"
]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
data = data.dropna()

# Prepare features and labels
X = data.drop(['Label'], axis=1)
y = data['Label'].astype(int)  # Convert labels to integers

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Reduced Feature Dimension: {X_train_pca.shape[1]}")

# Build the model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_pca.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    np.expand_dims(X_train_pca, axis=-1), y_train,
    validation_data=(np.expand_dims(X_test_pca, axis=-1), y_test),
    epochs=50, batch_size=32, callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(np.expand_dims(X_test_pca, axis=-1), y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions
predictions = (model.predict(np.expand_dims(X_test_pca, axis=-1)) > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
model.save('emotion_detection_model.h5')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() 