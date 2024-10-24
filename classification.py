import numpy as np  # Add this import
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Build the CNN model using transfer learning (VGG16)
def build_cnn_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model with early stopping and learning rate reduction
def train_cnn_model(model, image_generator, epochs=30):
    early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=3, min_lr=0.00001)
    
    history = model.fit(image_generator, epochs=epochs, callbacks=[early_stopping, reduce_lr])
    return history

# Save the trained model to a file
def save_cnn_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Load the trained model from a file
def load_cnn_model(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model not found at {model_path}")
        return None

# Predict for a new image using the trained CNN model
def predict_image(cnn_model, image):
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input
    prediction = cnn_model.predict(image)
    
    if prediction < 0.5:
        return "Benign"
    else:
        return "Malignant"
