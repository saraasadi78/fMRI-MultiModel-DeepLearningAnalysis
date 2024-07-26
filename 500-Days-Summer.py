import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta

# GPU setup
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
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
pickle_file = os.path.join(base_path, "500-Days-Summer.pickle")

def count_files(folder):
    file_count = 0
    for subject_folder in os.listdir(os.path.join(base_path, folder)):
        subject_path = os.path.join(base_path, folder, subject_folder)
        if os.path.isdir(subject_path):
            for file in os.listdir(subject_path):
                if file.endswith('_masked.txt'):
                    file_count += 1
    return file_count

def load_data(folder, start, end):
    data = []
    subject_count = 0
    sorted_subject_folders = sorted(os.listdir(os.path.join(base_path, folder)))
    
    for subject_folder in sorted_subject_folders:
        if subject_count >= end:
            break
        subject_path = os.path.join(base_path, folder, subject_folder)
        if os.path.isdir(subject_path) and subject_folder.startswith('sub-'):
            subject_number = int(subject_folder.split('-')[1])
            if start <= subject_number <= end:
                for file in os.listdir(subject_path):
                    if file.endswith('_masked.txt'):
                        data.append(np.loadtxt(os.path.join(subject_path, file), delimiter=','))
                        print(subject_folder, "done")
                        subject_count += 1

    print(f"Processed {subject_count} subjects from {folder}.")
    return data

# Count and load data
face_file_count = count_files('face')
noface_file_count = count_files('noface')
print(f"Number of '_masked.txt' files in 'face': {face_file_count}")
print(f"Number of '_masked.txt' files in 'noface': {noface_file_count}")

face_data = load_data('face', 1, 20)   
noface_data = load_data('noface', 1, 20)

X_data = np.array(face_data + noface_data)
y_data = np.array([1] * len(face_data) + [0] * len(noface_data))

height, width = X_data.shape[1], X_data.shape[2]
X_data = X_data.reshape(-1, height, width, 1)

print(y_data)
print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")
print(f"X_data size (in bytes): {X_data.nbytes}")

# Save data to pickle file
print("Saving data to pickle file...")
with open(pickle_file, 'wb') as f:
    pickle.dump({'X': X_data, 'y': y_data}, f)

# Load data from pickle file
print("Loading data from pickle file...")
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    X_data, y_data = data['X'], data['y']
else:
    print("Pickle file not found. Please check the file path.")
    exit()

print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, shuffle=True, random_state=42)

print("Class distribution in training set:", Counter(y_train))
print("Class distribution in test set:", Counter(y_test))

X_train_reshaped = X_train.reshape(X_train.shape[0], height, width, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], height, width, 1)


model = Sequential([
    Conv2D(16, (3, 3), activation='tanh', input_shape=(height, width, 1)),
    Conv2D(32, (3, 3), activation='tanh'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='tanh'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adadelta(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam (learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) 
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) 


early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

with tf.device('/device:GPU:0'):
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test_reshaped, y_test),
        callbacks=[early_stopping]
    )


loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test accuracy: {accuracy}')

model.summary()
