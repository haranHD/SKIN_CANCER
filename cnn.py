import os
from process import process_image, setup_data_generator
from classification import build_cnn_model, train_cnn_model, save_cnn_model, load_cnn_model, predict_image

# Global variables for file paths
metadata_path = r'C:\Users\haran\Downloads\HAM10000_metadata.csv'  # Metadata file path
image_folder = r'C:\Users\haran\Downloads\HAM10000_classified_images'  # Parent folder path
model_path = r'C:\Users\haran\Downloads\skin_cancer_cnn_model.h5'  # Model save path

# Main function to execute all tasks
def main():
    # Check if the model exists and load it if available
    cnn_model = load_cnn_model(model_path)
    
    if cnn_model is None:
        # Setup image data generator
        print("Setting up Image Data Generator:")
        image_generator = setup_data_generator(image_folder, batch_size=64)

        # Build and train the CNN model if not found
        print("Building and training CNN model:")
        cnn_model = build_cnn_model(input_shape=(128, 128, 3))
        cnn_model.summary()
        history = train_cnn_model(cnn_model, image_generator, epochs=30)
        
        # Save the trained model
        save_cnn_model(cnn_model, model_path)
    
    # Repeat image input until user says 'over'
    while True:
        image_id = input("Enter the image ID (or type 'over' to stop): ")

        if image_id.lower() == 'over':
            print("Process terminated.")
            break

        # Decide subfolder ('benign' or 'malignant') based on your image classification
        subfolder = input("Enter the subfolder (benign/malignant): ").strip().lower()

        # Process and predict the class of the image
        processed_image = process_image(image_folder, image_id, subfolder)
        prediction = predict_image(cnn_model, processed_image)

        # Display prediction result
        print(f"Prediction for {image_id}: {prediction}")

# Run the main function
if __name__ == "__main__":
    main()
