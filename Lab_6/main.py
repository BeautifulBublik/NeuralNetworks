import os
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt



logo_dir = 'logos'
background_dir = 'background'
output_dir = 'Dataset'

splits = ['train', 'val', 'test']
classes = ['positive', 'negative']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

def augment_logo(logo):
    angle = random.uniform(-30, 30)
    logo = logo.rotate(angle, expand=True)
    scale = random.uniform(0.5, 1.5)
    w, h = logo.size
    logo = logo.resize((int(w * scale), int(h * scale)))
    alpha = random.uniform(0.5, 1.0)
    logo.putalpha(int(255 * alpha))
    return logo

logo_files = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if
f.endswith(('.png', '.jpg', '.jpeg'))]
background_files = [os.path.join(background_dir, f) for f in
os.listdir(background_dir) if f.endswith(('.png', '.jpg','.jpeg'))]

total_positive = 1000
total_negative = 1000
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def get_split_name(idx, total):
    val_thresh = int(total * split_ratios['train' ])
    test_thresh = int(total * (split_ratios['train'] + split_ratios['val']))
    if idx < val_thresh:
        return 'train'
    elif idx < test_thresh:
        return 'val'
    else:
        return 'test'
i = 0
pbar = tqdm(total=total_positive, desc='Generating positive samples')
while i < total_positive:
    bg_path = random.choice(background_files)
    logo_path = random.choice(logo_files)

    try:
        bg = Image.open(bg_path).convert('RGB')
        logo = Image.open(logo_path).convert('RGBA')
    except Exception as e:
        print(f"Error opening image: {e}")
        continue

    logo = augment_logo(logo)

    if logo.width > bg.width or logo.height > bg.height:
        logo.thumbnail((bg.width, bg.height))

    max_x = bg.width - logo.width
    max_y = bg.height - logo.height

    if max_x < 0 or max_y < 0:
        continue

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    bg.paste(logo, (x, y), logo)

    split = get_split_name(i, total_positive)
    save_path = os.path.join(output_dir, split, 'positive', f'pos_{i}.jpg')
    bg.save(save_path)

    i += 1
    pbar.update(1)

pbar.close()

i = 0
pbar = tqdm(total=total_negative, desc='Generating negative samples')

while i < total_negative:
    bg_path = random.choice(background_files)

    try:
        bg = Image.open(bg_path).convert('RGB')
    except Exception as e:
        print(f"Error opening background image: {e}")
        continue

    split = get_split_name(i, total_negative)
    save_path = os.path.join(output_dir, split, 'negative', f'neg_{i}.jpg')
    bg.save(save_path)

    i += 1
    pbar.update(1)

pbar.close()

dataset_dir = 'Dataset'
img_size = (200, 200)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen. flow_from_directory(
    directory=os.path. join(dataset_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen. flow_from_directory(
    directory=os.path. join(dataset_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen. flow_from_directory(
    directory=os.path. join(dataset_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

def build_xception(input_shape=(150, 150, 3), num_classes=1):
    inputs = layers. Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    skip = layers.Conv2D(32, (1,1), strides=(2,2),padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.SeparableConv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2,2),padding='same')(x)

    x = layers.add([x, skip])

    skip = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers. SeparableConv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2,2), padding='same')(x)

    x = layers.add([x, skip])

    x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

model = build_xception()
early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1)
history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[early_stop] )

print("\nEvaluating on test data...")
test_logits = model.predict(test_generator)
test_probs = tf.nn.sigmoid(test_logits).numpy().flatten()
test_preds = (test_probs > 0.5).astype(int)
test_labels = test_generator.labels
test_acc = np.mean(test_preds == test_labels)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test prediction stats:")
print(f" Min: {test_probs.min():.4f}, Max: {test_probs.max():.4f}")
print(f" Mean: {test_probs.mean():.4f}, Std: {test_probs.std():.4f}")

plt.figure(figsize=(10, 5))
plt.hist(test_probs, bins=20, alpha=0.7)
plt.title('Test Predictions Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.savefig('test_predictions.png')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout ()
plt.savefig('training_history.png')

num_samples = 5
class_names = list(train_generator.class_indices.keys())
num_test_samples = len(test_generator. labels)
random_indices = random. sample(range(num_test_samples), num_samples)
plt.figure(figsize=(15, 10))
for i, index in enumerate(random_indices):
    image = test_generator.filepaths[index]
    true_label = test_generator.labels[index]
    img = tf.keras.utils.load_img(image, target_size=model.input_shape[1:3])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    probabilities = tf.nn.sigmoid(predictions).numpy(). flatten()
    predicted_class = 1 if probabilities[0] > 0.5 else 0
    predicted_label = class_names[predicted_class]
    true_label_name = class_names[true_label]

    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label_name}\nPredicted: {predicted_label} ({probabilities[0]:.2f})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
base_path = './'
file_name = f'xception_{timestamp}.keras'
model_save_path = base_path + file_name
model.save(model_save_path)
print(f"Модель успішно збережено за шляхом: {model_save_path}")



