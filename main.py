# TensorFlow and tf.keras
import tensorflow as tf

# Utils
import os

# Configuration
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 10
EPOCHS_NUM = 40
print("Using Tensorflow", tf.__version__)


def create_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes))

    return model


def load_dataset(path):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    )
    return dataset


if __name__ == "__main__":
    # create class names
    for folder in os.listdir("./data"):
        print(folder)
        with open("./output/descriptions.txt", "a+") as f:
            f.write(folder + "\n")

    # Load the dataset
    data = load_dataset("./data")

    # Create the model
    model = create_model(len(data.class_names))
    model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(data, epochs=EPOCHS_NUM, validation_data=data)

    # Save the model
    model.save("output/keras/model_" + str(EPOCHS_NUM) + ".h5")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(data)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # End
    print("Dataset loaded and model trained")
