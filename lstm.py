import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from collections import Counter
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

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
        print(e)

# Constants
BASE_PATH = r"E:\Users\sara.asadi\Desktop\pypro"
PICKLE_FILE = os.path.join(BASE_PATH, "fmri_data.pickle")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TIME_STEPS = 10
FEATURES = 90240
CLIP_VALUE = 1.0
EPOCHS = 100
BATCH_SIZE = 24
PATIENCE = 20


if os.path.exists(PICKLE_FILE):
    print("Loading data from pickle file...")
    with open(PICKLE_FILE, 'rb') as f:
        data = pickle.load(f)
    X_data, y_data = data['X'], data['y']
else:
    print("Pickle file not found. Please check the file path.")
    exit()


scaler = StandardScaler()
X_data = scaler.fit_transform(X_data.reshape(-1, FEATURES)).reshape(-1, TIME_STEPS, FEATURES)
X_data = X_data.squeeze(axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, stratify=y_data, shuffle=True, random_state=RANDOM_STATE)
X_train_moved = np.moveaxis(X_train, 1, 2)
X_test_moved = np.moveaxis(X_test, 1, 2)


model = Sequential([
    LSTM(64, activation='tanh', input_shape=(TIME_STEPS, FEATURES), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model2 = Sequential([
    Bidirectional(LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(TIME_STEPS, FEATURES)),
    Dropout(0.2),
    Bidirectional(LSTM(32, activation='tanh', kernel_regularizer=l2(0.01))),
    Dropout(0.2),
    Dense(32, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.001)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)


optimizer = RMSprop()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')


with tf.device('/device:GPU:0'):
    history = model.fit(
        X_train_moved, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_moved, y_test),
        callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard]

    )


loss, accuracy = model.evaluate(X_test_moved, y_test)
print(f'Test accuracy: {accuracy}')

# Classification report
y_pred = (model.predict(X_test_moved) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


model.summary()

# Plot accuracy and loss
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


'''
model = Sequential([
    Bidirectional(LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(TIME_STEPS, FEATURES)),
    Dropout(0.2),
    Bidirectional(LSTM(64, activation='relu', kernel_regularizer=l2(0.01))),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


optimizer = Adam(lr=0.001, decay=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
'''




