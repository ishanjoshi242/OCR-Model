import tensorflow as tf
import cv2
import numpy as np
import plotly.graph_objects as go

# Load the OCR model
model_path = 'C:/Users/Ishan/project1/ocr_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the model for CPU inference
with tf.device('/CPU:0'):
    model_cpu = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format for optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model_cpu)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the optimized model
with open('model_cpu_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

# Set input dimensions for the model
input_shape = model.input_shape
height, width, channels = input_shape[1], input_shape[2], input_shape[3]

# Load the video file
video_path = 'C:/Users/Ishan/project1/OCR Alphabet.mp4'
cap = cv2.VideoCapture(video_path)

# Preprocessing function for each frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (width, height))
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Lists to store FPS values for both GPU and CPU
fps_gpu = []
fps_cpu = []

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the current frame
    processed_frame = preprocess_frame(frame)

    # Measure GPU prediction time
    start_time_gpu = cv2.getTickCount()
    prediction_gpu = model.predict(processed_frame)
    end_time_gpu = cv2.getTickCount()
    time_elapsed_gpu = (end_time_gpu - start_time_gpu) / cv2.getTickFrequency()
    fps_gpu.append(1 / time_elapsed_gpu)

    # Measure CPU prediction time
    start_time_cpu = cv2.getTickCount()
    prediction_cpu = model_cpu.predict(processed_frame)
    end_time_cpu = cv2.getTickCount()
    time_elapsed_cpu = (end_time_cpu - start_time_cpu) / cv2.getTickFrequency()
    fps_cpu.append(1 / time_elapsed_cpu)

    # Display the prediction on the frame
    prediction_text = str(prediction_cpu)
    cv2.putText(frame, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('OCR Output', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Plotting the FPS comparison using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(fps_gpu))), y=fps_gpu, mode='lines+markers', name='GPU'))
fig.add_trace(go.Scatter(x=list(range(len(fps_cpu))), y=fps_cpu, mode='lines+markers', name='CPU'))
fig.update_layout(title='FPS Comparison: GPU vs CPU', xaxis_title='Frame', yaxis_title='FPS')
fig.show()
