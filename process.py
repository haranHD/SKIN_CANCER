import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Process an image: Resize and normalize using OpenCV
def process_image(image_folder, image_name, subfolder, target_size=(128, 128)):
    image_path = os.path.join(image_folder, subfolder, image_name + '.jpg')
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

# Set up image data generator for augmentations
def setup_data_generator(image_folder, batch_size=64):
    datagen = ImageDataGenerator(
        rescale=1./255,          
        rotation_range=30,       
        width_shift_range=0.2,   
        height_shift_range=0.2,  
        shear_range=0.2,         
        zoom_range=0.2,          
        horizontal_flip=True,    
        fill_mode='nearest'      
    )
    
    image_generator = datagen.flow_from_directory(
        image_folder,
        target_size=(128, 128),  
        batch_size=batch_size,   
        class_mode='binary'      
    )
    
    return image_generator
