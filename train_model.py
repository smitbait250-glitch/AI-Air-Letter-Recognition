import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

print("Loading EMNIST Letters dataset...")

# Load EMNIST Letters
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# ------------------------------------------------
# Prepare training data
# ------------------------------------------------

def prepare_data(image, label):

    # Convert image from uint8 to float
    image = tf.cast(image, tf.float32) / 255.0

    # EMNIST letters labels are 1-26
    # Convert them to 0-25
    label = label - 1

    return image, label


ds_train = ds_train.map(
    prepare_data,
    num_parallel_calls=tf.data.AUTOTUNE
)

ds_test = ds_test.map(
    prepare_data,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Shuffle and batch
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------
# Create CNN model
# ------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        26,
        activation="softmax"
    )
])

# ------------------------------------------------
# Compile
# ------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------
# Train
# ------------------------------------------------

print("\nStarting training...\n")

model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test
)

# ------------------------------------------------
# Save model
# ------------------------------------------------

os.makedirs("models", exist_ok=True)

model.save(
    "models/letter_model.keras"
)

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")
print("Model saved to:")
print("models/letter_model.keras")