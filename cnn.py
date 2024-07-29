import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, PReLU, ELU, ReLU, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Nadam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

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

 
base_path = r"E:\Users\sara.asadi\Desktop\pypro"
pickle_file = os.path.join(base_path, "fmri_data.pickle")
preprocessed = os.path.join(base_path, "preprocessedallsubjects.pickle")
Days_Summer= os.path.join(base_path, "500-Days-Summer.pickle")
avgdata= os.path.join(base_path, "avgdata.pickle")


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
 
 
height, width = X_data.shape[1], X_data.shape[2]
 
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data,shuffle=True, random_state=42)

 
print("Class distribution in training set:", Counter(y_train))
print("Class distribution in test set:", Counter(y_test))


X_train_reshaped = X_train.reshape(X_train.shape[0], height, width, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], height, width, 1)

X_input = X_train_reshaped
X_test=X_test_reshaped


print("\nSample data:")
print(f"X_train[0] shape: {X_train[0].shape}")
print(f"X_train[0] min: {X_train[0].min()}, max: {X_train[0].max()}")
print(f"y_train[0]: {y_train[0]}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)),
    BatchNormalization(),
    #MaxPooling2D((2, 2)),
    AveragePooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    #BatchNormalization(),
    MaxPooling2D((2, 2)),
    #AveragePooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    #BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']) 
#model.compile(optimizer=Adam (learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) 
#model.compile(optimizer=Adadelta(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) 

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

with tf.device('/device:GPU:0'):
    history = model.fit(
        X_input, y_train,
        epochs= 150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=early_stopping
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
