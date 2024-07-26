import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)


base_path = r"E:\Users\sara.asadi\Desktop\pypro"
preprocessed = os.path.join(base_path, "preprocessedallsubjects.pickle")

if os.path.exists(preprocessed):
    print("Loading data from pickle file...")
    with open(preprocessed, 'rb') as f:
        data = pickle.load(f)
    X_data, y_data = data['X'], data['y']
else:
    print("Pickle file not found. Please check the file path.")
    exit()

print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")

# Reshape and scale 
height, width = X_data.shape[1], X_data.shape[2]
X_data_reshaped = X_data.reshape(-1, height * width)
scaler = StandardScaler()
X_data_scaled = scaler.fit_transform(X_data_reshaped)
X_data_scaled = X_data_scaled.reshape(-1, height, width, 1)

X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data, test_size=0.2, stratify=y_data, random_state=42)

print("Class distribution in training set:", Counter(y_train))
print("Class distribution in test set:", Counter(y_test))

def create_model(learning_rate=0.0001, kernel_regularizer=l2(0.01), dropout_rate=0.5):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer, input_shape=(height, width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=kernel_regularizer),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# KerasClassifier for GridSearchCV
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

param_grid = {
    'learning_rate': [0.001, 0.0001],
    'kernel_regularizer': [l2(0.01), l2(0.1)],
    'dropout_rate': [0.3, 0.5]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=KFold(n_splits=5, shuffle=True, random_state=42))
grid_result = grid.fit(X_train, y_train)

# Print grid search results
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best accuracy found: {grid_result.best_score_}")

# Final model with best parameters
best_model = grid_result.best_estimator_

# Final model training on the full training set
final_history = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)],
    class_weight=compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
)


loss, accuracy = best_model.score(X_test, y_test)
print(f'Test accuracy: {accuracy}')

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(final_history.history['accuracy'], label='Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(final_history.history['loss'], label='Training Loss')
plt.plot(final_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ROC curve and classification report
y_pred_proba = best_model.predict_proba(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")

y_pred_classes = (y_pred_proba > optimal_threshold).astype(int)

print(classification_report(y_test, y_pred_classes, zero_division=1))

# ROC curve plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

'''
# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    model = create_model()
    
    history = model.fit(
        X_train_fold, y_train_fold,
        batch_size=32,
        epochs=50,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )
    
    score = model.evaluate(X_val_fold, y_val_fold)
    cv_scores.append(score[1])
    print(f"Fold {fold + 1} accuracy: {score[1]}")

print(f"Mean CV accuracy: {np.mean(cv_scores)}")
'''
