from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_data_gen(filepath):
    # Initializing Training ImageDataGenerator class with various augmentation parameters
    train_datagen = ImageDataGenerator(
        rescale=1./255.,  # Rescale pixel values to the range [0, 1]
        rotation_range=40,  # Random rotation within the range [-40, 40] degrees
        width_shift_range=0.2,  # Random horizontal shift within the range [-0.2, 0.2] of the width
        height_shift_range=0.2,  # Random vertical shift within the range [-0.2, 0.2] of the height
        shear_range=0.2,  # Shear intensity within the range [-0.2, 0.2]
        zoom_range=0.2,  # Random zoom within the range [0.8, 1.2]
        horizontal_flip=True  # Random horizontal flipping
    )

    # Create a data generator for training from the specified directory
    train_generator = train_datagen.flow_from_directory(
        filepath,
        batch_size=32,
        class_mode='binary',  # Classification mode (binary in this case)
        target_size=(150, 150),  # Resize images to (150, 150)
        seed=99  # Random seed for reproducibility
    )

    return train_generator

def get_validation_data_gen(filepath):
    # Initializing validation `ImageDataGenerator` class with only rescale parameter.
    # Other augmentation parameters are set to default.
    validation_datagen = ImageDataGenerator(rescale=1./255.)

    # Create a data generator for validation from the specified directory
    validation_generator = validation_datagen.flow_from_directory(
        filepath,
        shuffle=False,  # Don't shuffle the data (important for evaluation)
        batch_size=32,
        class_mode='binary',  # Classification mode (binary in this case)
        target_size=(150, 150)  # Resize images to (150, 150)
    )

    return validation_generator
