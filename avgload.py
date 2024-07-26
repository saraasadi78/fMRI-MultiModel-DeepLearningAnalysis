import os
import numpy as np
import tensorflow as tf
import pickle

base_path = r"E:\Users\sara.asadi\Desktop\pypro"

def load_data(folder, limit=None):
    data = []
    subject_count = 0
    for subject_folder in os.listdir(os.path.join(base_path, folder)):
        if limit and subject_count >= limit:
            break
        subject_path = os.path.join(base_path, folder, subject_folder)
        if os.path.isdir(subject_path):
            for file in os.listdir(subject_path):
                if file.endswith('.txt') and ('average' in file):
                    file_path = os.path.join(subject_path, file)
                    try:
                        # Read the file as text
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Process each line, keeping only numeric values
                        numeric_data = []
                        for line in lines:
                            try:
                                # Split the line by comma and convert to float
                                row = [float(x) for x in line.strip().split(',') if x.strip()]
                                if row:  # Only append non-empty rows
                                    numeric_data.append(row)
                            except ValueError:
                                # If conversion fails, skip this line
                                continue
                        
                        if numeric_data:
                            data.append(np.array(numeric_data))
                            print(f"{subject_folder}, {file} done")
                        else:
                            print(f"No valid numeric data in {file} from {subject_folder}")
                    
                    except Exception as e:
                        print(f"Error processing {file} in {subject_folder}: {e}")
                        continue
            subject_count += 1
    print(f"Processed {subject_count} subjects from {folder}.")
    return data


face_data = load_data('face')
noface_data = load_data('noface')


X_data = np.array(face_data + noface_data)
y_data = np.array([1] * len(face_data) + [0] * len(noface_data))


if X_data.size == 0:
    print("No data was loaded. Please check your file contents and paths.")
    exit()

# Reshape data
try:
    height, width = X_data.shape[1], X_data.shape[2]
    X_data = X_data.reshape(-1, height, width, 1)
except IndexError:
    print("Error: Data shape is not as expected. Please check the loaded data.")
    print(f"X_data shape: {X_data.shape}")
    exit()


print(y_data)
print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")
print(f"X_data size (in bytes): {X_data.nbytes}")

pickle_file = os.path.join(base_path, "avgdata.pickle")
print("Saving data to pickle file...")
with open(pickle_file, 'wb') as f:
    pickle.dump({'X': X_data, 'y': y_data}, f)

print("Data saved successfully.")
