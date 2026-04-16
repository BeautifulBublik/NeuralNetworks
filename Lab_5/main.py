import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = "dataset"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding,
                      use_bias=False, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.Activation('relu')(x)
    return x
def build_manual_inception_v3(num_classes=2):
    inputs = layers.Input(shape=(299, 299, 3))

    x = data_augmentation(inputs)
    x = conv2d_bn(x, 16, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 16, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    branch1x1 = conv2d_bn(x, 32, 1, 1)

    branch5x5 = conv2d_bn(x, 32, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5)

    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.Concatenate(axis=3)([branch1x1, branch5x5, branch3x3dbl, branch_pool])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="micro_inception_scratch")

my_model = build_manual_inception_v3(num_classes=2)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

y_train = np.concatenate([y for x, y in train_ds], axis=0)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = my_model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

loss, accuracy = my_model.evaluate(val_ds)
print(f"Точність на валідації: {accuracy:.4f}")
print(f"Втрати (Loss): {loss:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

def predict_image(model, test_dir, class_names):
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            img = image.load_img(img_path, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array, verbose=0)
            pred_class = class_names[np.argmax(preds)]

            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Actual: {class_name} | Predicted: {pred_class}")
            plt.show()

predict_image(my_model, "test", class_names)