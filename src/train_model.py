import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data/train"

# DATA GENERATOR
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# MODEL (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# COMPILE
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# SAVE MODEL
model.save("model/keras_model.h5")

# SAVE LABELS
labels = list(train_data.class_indices.keys())

with open("model/labels.txt", "w") as f:
    for i, label in enumerate(labels):
        f.write(f"{i} {label}\n")

print("✅ Training Complete!")