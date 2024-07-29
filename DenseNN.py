import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import pickle

# GPU setup
print(tf.config.list_physical_devices('GPU'))
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print("Error setting up GPU:", e)


base_path = r"C:\Users\sara.asadi\Desktop\pypro"
pickle_file = os.path.join(base_path, "fmri_data.pickle")

if os.path.exists(pickle_file):
    print("Loading data from pickle file...")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    X_data, y_data = data['X'], data['y']
else:
    print("Pickle file not found. Please check the file path.")
    exit()

print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")


X_data = X_data.squeeze(axis=-1)
print(f"X_data shape after removing last dimension: {X_data.shape}")

X_data_2d = X_data.reshape(X_data.shape[0], -1)

scaler = StandardScaler()
X_data_scaled = scaler.fit_transform(X_data_2d)

# PCA
n_components = min(X_data_scaled.shape[0], X_data_scaled.shape[1], 1000)
pca = PCA(n_components=n_components)
X_data_pca = pca.fit_transform(X_data_scaled)

print(f"Shape after PCA: {X_data_pca.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_data_pca, y_data, test_size=0.2, stratify=y_data, shuffle=True, random_state=42)

print("Class distribution in training set:", np.bincount(y_train))
print("Class distribution in test set:", np.bincount(y_test))


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


optimizer = Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000001)


history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model.summary()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
