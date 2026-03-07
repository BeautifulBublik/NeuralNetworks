from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import os


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    x_train,
    y_train_cat,
    epochs=30,
    batch_size=32,
    validation_data=(x_test, y_test_cat)
)
loss, accuracy = model.evaluate(x_test, y_test_cat)

print("\nAccuracy:", accuracy)

plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")


print("\nClassification report:")
print(classification_report(y_test, y_pred_classes))

plt.show()

def recognize_digit(path):
    img = Image.open(path).convert('L')
    img = img.resize((28,28))
    img = np.array(img)
    img = 255 - img
    img = img.astype("float32") / 255
    img = img.reshape(1,784)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    return digit, img.reshape(28,28)
def recognize_array(image):
    image = image.reshape(1, 784)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return digit

folder = "my_digits"
print("\nРозпізнавання власноруч написаних цифр:\n")
for file in os.listdir(folder):
    path = os.path.join(folder, file)
    digit, img = recognize_digit(path)
    print(file, "->", digit)
    plt.imshow(img, cmap="gray")
    plt.title(f"{file} -> {digit}")
    plt.axis("off")
    plt.show()

for i in range(5):
    img = x_test[i].reshape(28, 28)
    digit = recognize_array(img)
    plt.imshow(img, cmap="gray")
    plt.title(f"Розпізнано: {digit}")
    plt.axis("off")
    plt.show()



