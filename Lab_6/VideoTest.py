import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model_path = 'xception_20260418_230727.keras'
video_path = 'Mercedes2.mp4'
model = load_model(model_path)
print("Модель успішно завантажено!")
cap = cv2.VideoCapture(video_path)
batch_size = 32
frames = []
frames_ids = []
frame_predictions = []
frame_numbers = []
frame_count = 0
def process_batch(batch_frames, batch_ids):

    batch_np = np.array(batch_frames).astype(np.float32)
    batch_np = batch_np / 255.0
    predictions = model.predict(batch_np)

    for i in range(len(predictions)):
        score = float(predictions[i][0])
        # score = 1 / (1 + np.exp(-predictions[i][0]))
        frame_predictions.append(score)
        frame_numbers.append(batch_ids[i])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (150, 150))
    frames.append(resized_frame)
    frames_ids.append(frame_count)

    if len(frames) == batch_size:
        process_batch(frames, frames_ids)
        frames = []
        frames_ids = []
    frame_count += 1
if frames:
    process_batch(frames, frames_ids)
cap.release()
plt.figure(figsize=(12, 6))
plt.plot(frame_numbers, frame_predictions, label='Confidence', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.xlabel('Frame Number')
plt.ylabel('Prediction Score')
plt.title('Model Predictions Over Video Frames')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()