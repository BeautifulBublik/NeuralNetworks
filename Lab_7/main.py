import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


from nltk.corpus import stopwords
nltk.download("stopwords")
print("Loading Yelp review data..." )
stop_words_set = set(stopwords.words('english'))

train_path = "yelp_review_polarity_csv/yelp_review_polarity_csv/train.csv"
test_path = "yelp_review_polarity_csv/yelp_review_polarity_csv/test.csv"

df_train = pd.read_csv(train_path, names=['label', 'text'])
df_test = pd.read_csv(test_path, names=['label', 'text'])

df_train['label'] = df_train['label'] - 1
df_test['label'] = df_test['label'] - 1

df_train = df_train.sample(n=40000, random_state=42)
df_test = df_test.sample(n=10000, random_state=42)

print(f"Successfully loaded {len(df_train)} train and {len(df_test)} test entries")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words_set])
    return text

print("Cleaning text (this might take a minute)...")
X_train_clean = df_train['text'].apply(preprocess_text)
y_train = df_train['label'].values

X_test_clean = df_test['text'].apply(preprocess_text)
y_test = df_test['label'].values

print("Converting text to sequences...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train_clean)

train_sequences = tokenizer.texts_to_sequences(X_train_clean)
test_sequences = tokenizer.texts_to_sequences(X_test_clean)

sequence_length = 100
X_train_pad = pad_sequences(train_sequences, maxlen=sequence_length, padding='post')
X_test_pad = pad_sequences(test_sequences, maxlen=sequence_length, padding='post')

print("Tokenization and padding completed")

print("Constructing LSTM model...")
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training the model...")
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))
print("Evaluating on test data...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Set Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='ValidationAccuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

print("Generating predictions for the test set...")
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int)
y_pred = y_pred.flatten()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-Score:   {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])

plt.title('Confusion Matrix')
plt.ylabel('Actual (Справжній клас)')
plt.xlabel('Predicted (Передбачений клас)')
plt.show()
def analyze_sentiment(input_text):
    cleaned = preprocess_text(input_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=sequence_length, padding='post')
    prob = model.predict(padded)[0][0]
    return "Positive" if prob > 0.5 else "Negative"


print("\n--- Sample prediction tests ---")

test_reviews = [
    "The best pizza in town! The crust was crispy, toppings were fresh, and the staff was super friendly.",
    "Terrible experience. The soup was cold, the waiter was rude, and it took an hour to get the bill.",
    "Highly recommend! Fast service and delicious coffee.",
    "Do not eat here. Found a hair in my pasta and the manager didn't even apologize. Save your money.",
    "The ambiance is beautiful and the drinks are okay, but the food is ridiculously overpriced for such small portions."
]

for i, review in enumerate(test_reviews, start=1):
    sentiment = analyze_sentiment(review)
    print(f"Sample {i}:")
    print(f"Review: '{review}'")
    print(f"Predicted Sentiment: {sentiment}\n")