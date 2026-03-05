import tensorflow as tf
import numpy as np

print("\n=== XOR для 3 входів ===")
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_dim=3, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100)

loss, accuracy = model.evaluate(x, y)
print("loss", loss)
print("accuracy", accuracy)

prediction = model.predict(x)
for inp, pred in zip(x, prediction):
    print(inp, round(pred[0]))

print("\n=== XOR для 4 входів ===")

x4 = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
])

y4 = np.array([0, 1,1,0,1,0,0,1,1,0,0,1,0,1,1,0])


model4 = tf.keras.Sequential([
    tf.keras.layers.Dense(8,input_dim=4 ,activation="tanh", ),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model4.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy']
)
model4.fit(x4, y4, epochs=150)

loss4, accuracy4 = model4.evaluate(x4, y4)
print("loss:", loss4)
print("accuracy:", accuracy4)

predictions4 = model4.predict(x4)
for inp, pred in zip(x4, predictions4):
    print(inp, round(pred[0]))
