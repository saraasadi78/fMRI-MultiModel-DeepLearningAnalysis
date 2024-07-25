import os
import numpy as np
import tensorflow as tf
import pickle

base_path = r"E:\Users\sara.asadi\Desktop\pypro"

face_data = []
noface_data = []

def count_files(folder):
    file_count = 0
    for subject_folder in os.listdir(os.path.join(base_path, folder)):
        subject_path = os.path.join(base_path, folder, subject_folder)
        if os.path.isdir(subject_path):
            for file in os.listdir(subject_path):
                if file.endswith('_masked.txt'):
                    file_count += 1
    return file_count

def load_data(folder, limit=None):
    data = []
    subject_count = 0
    for subject_folder in os.listdir(os.path.join(base_path, folder)):
        if limit and subject_count >= limit:
            break
        subject_path = os.path.join(base_path, folder, subject_folder)
        if os.path.isdir(subject_path):
            for file in os.listdir(subject_path):
                if file.endswith('_masked.txt'):
                    data.append(np.loadtxt(os.path.join(subject_path, file), delimiter=','))
                    print(subject_folder, "done")
            subject_count += 1
    print(f"Processed {subject_count} subjects from {folder}.")
    return data

face_file_count = count_files('face')
noface_file_count = count_files('noface')

print(f"Number of '_masked.txt' files in 'face': {face_file_count}")
print(f"Number of '_masked.txt' files in 'noface': {noface_file_count}")

face_data = load_data('face')
noface_data = load_data('noface')

X_data = np.array(face_data + noface_data)
y_data = np.array([1] * len(face_data) + [0] * len(noface_data))

height, width = X_data.shape[1], X_data.shape[2]
X_data = X_data.reshape(-1, height, width, 1)

print(y_data)

print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")
print(f"X_data size (in bytes): {X_data.nbytes}")

pickle_file = os.path.join(base_path, "fmri_data.pickle")

print("Saving data to pickle file...")
with open(pickle_file, 'wb') as f:
    pickle.dump({'X': X_data, 'y': y_data}, f)

