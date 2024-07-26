import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import argparse

base_path = r"E:\Users\sara.asadi\Desktop\pypro"
pickle_file = os.path.join(base_path, "fmri_data.pickle")
Days_Summer = os.path.join(base_path, "500-Days-Summer.pickle")

def load_data(file_path):
    """Load data from pickle file."""
    if os.path.exists(pickle_file):
        print("Loading data from pickle file...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']
    else:
        raise FileNotFoundError(f"Pickle file not found at {file_path}")


def preprocess_data(X, y, n_components=1000, test_size=0.2, random_state=42):
    """Preprocess the data: scale, apply PCA, and split into train/test sets."""
    # Reshape data to 2D for preprocessing
    X_reshaped = X.reshape(X.shape[0], -1)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Apply PCA
    n_components = min(X_scaled.shape[0], X_scaled.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Shape after PCA: {X_pca.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Split the data into training and testing sets
    return train_test_split(X_pca, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the classes in the training data."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def reshape_for_cnn(X, timesteps):
    """Reshape data for CNN model."""
    features = X.shape[1] 
    return X.reshape(-1, timesteps, features, 1)

def reshape_for_lstm(X, timesteps):
    """Reshape data for LSTM model."""
    features = X.shape[1]  // timesteps
    return X.reshape(-1, timesteps, features)

def save_preprocessed_data(output_path, data_dict):
    print(f"Saving preprocessed data to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

def main(input_file, output_dir, model_type, n_components=1000, timesteps=10):
    # Load data
    X, y = load_data(input_file)
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y, n_components)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    print(f"Training data shape after SMOTE: {X_train_resampled.shape}")

    # Prepare data based on model type
    if model_type == 'cnn':
        X_train_final = reshape_for_cnn(X_train_resampled, timesteps)
        X_test_final = reshape_for_cnn(X_test, timesteps)
    elif model_type == 'lstm':
        X_train_final = reshape_for_lstm(X_train_resampled, timesteps)
        X_test_final = reshape_for_lstm(X_test, timesteps)
    elif model_type == 'mlp':
        X_train_final = X_train_resampled
        X_test_final = X_test
    else:
        raise ValueError("Invalid model type. Choose 'cnn', 'lstm', or 'mlp'.")

    print(f"Final training data shape: {X_train_final.shape}")
    print(f"Final testing data shape: {X_test_final.shape}")

    # Save preprocessed data
    output_file = os.path.join(output_dir, f"preprocessed_{model_type}_data.pickle")
    save_preprocessed_data(output_file, {
        'X_train': X_train_final,
        'y_train': y_train_resampled,
        'X_test': X_test_final,
        'y_test': y_test
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess fMRI data for different model types.")
    parser.add_argument("input_file", help="Path to the input pickle file")
    parser.add_argument("output_dir", help="Directory to save the preprocessed data")
    parser.add_argument("model_type", choices=['cnn', 'lstm', 'mlp'], help="Type of model to preprocess for")
    parser.add_argument("--n_components", type=int, default=1000, help="Number of PCA components")
    parser.add_argument("--timesteps", type=int, default=10, help="Number of timesteps for CNN/LSTM reshaping")

    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.model_type, args.n_components, args.timesteps)



# Normalize the data to the range [0, 1]
data_min = np.min(X_data)
data_max = np.max(X_data)
normalized_data = (X_data - data_min) / (data_max - data_min)

X_data = normalized_data

max_value = np.max(X_data)
min_value = np.min(X_data)

print(f"XData max value: {max_value}")
print(f"XData min value: {min_value}")


print("Saving data to pickle file...")
with open(os.path.join(base_path, "preprocessed.pickle"), 'wb') as f:
    pickle.dump({'X': X_data, 'y': y_data}, f)

